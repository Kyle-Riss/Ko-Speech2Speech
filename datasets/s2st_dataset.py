"""
TSV-based dataset utilities for EchoStream.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
import numpy as np

try:
    import torchaudio
    from torchaudio.transforms import MelSpectrogram
    from torchaudio.functional import resample
except ImportError as exc:  # pragma: no cover - torchaudio should be installed
    raise ImportError(
        "torchaudio is required for EchoStream data loading. "
        "Please install it with `pip install torchaudio`."
    ) from exc


@dataclass
class ManifestEntry:
    """Single row of the S2ST manifest."""

    sample_id: str
    src_audio: Path
    src_text: str
    tgt_audio: Optional[Path]
    tgt_text: str
    tgt_units: Optional[Path] = None
    src_lang: Optional[str] = None
    tgt_lang: Optional[str] = None
    speaker: Optional[str] = None
    duration: Optional[float] = None


class TextTokenizer:
    """
    Minimal text tokenizer with optional vocabulary loading.

    The tokenizer supports word- or character-level tokenisation. If a
    vocabulary path is provided, the tokens are loaded from disk; otherwise the
    vocabulary grows dynamically based on observed text.
    """

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self, vocab_path: Optional[Path] = None, level: str = "word"):
        if level not in {"word", "char"}:
            raise ValueError(f"Unsupported tokenisation level: {level}")

        self.level = level
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        self.is_dynamic = vocab_path is None

        for token in [self.PAD, self.UNK, self.BOS, self.EOS]:
            self._add_token(token)

        if vocab_path is not None and Path(vocab_path).is_file():
            self.load_vocab(vocab_path)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS]

    def load_vocab(self, vocab_path: Path) -> None:
        with Path(vocab_path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                token = line.split()[0]
                if token in self.token_to_id:
                    continue
                self._add_token(token)

    def _add_token(self, token: str) -> int:
        idx = len(self.id_to_token)
        self.id_to_token.append(token)
        self.token_to_id[token] = idx
        return idx

    def update_from_corpus(self, texts: Iterable[str]) -> None:
        if not self.is_dynamic:
            return

        for text in texts:
            for token in self._tokenise(text):
                if token not in self.token_to_id:
                    self._add_token(token)

    def _tokenise(self, text: str) -> List[str]:
        if self.level == "char":
            return list(text.strip())
        return text.strip().split()

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)

        for token in self._tokenise(text):
            token_id = self.token_to_id.get(token)
            if token_id is None:
                if self.is_dynamic:
                    token_id = self._add_token(token)
                else:
                    token_id = self.unk_id
            tokens.append(token_id)

        if add_eos:
            tokens.append(self.eos_id)

        return tokens


class SpeechFeatureExtractor:
    """
    Extract log-mel spectrogram features from audio waveforms.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
        win_length: float = 0.025,
        hop_length: float = 0.010,
        n_fft: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        apply_log: bool = True,
    ):
        self.sample_rate = sample_rate
        self.apply_log = apply_log
        n_fft = n_fft or int(2 ** math.ceil(math.log2(sample_rate * win_length)))
        win_length_samples = int(sample_rate * win_length)
        hop_length_samples = int(sample_rate * hop_length)

        self.melspec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length_samples,
            hop_length=hop_length_samples,
            f_min=f_min,
            f_max=f_max,
            n_mels=num_mel_bins,
            center=True,
            power=2.0,
            normalized=False,
        )

        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=None) if apply_log else None

        self.hop_length_samples = hop_length_samples
        self.win_length_samples = win_length_samples

    def frame_shift(self) -> float:
        return float(self.hop_length_samples) / float(self.sample_rate)

    def __call__(self, waveform: torch.Tensor, orig_sample_rate: int) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if orig_sample_rate != self.sample_rate:
            waveform = resample(waveform, orig_freq=orig_sample_rate, new_freq=self.sample_rate)

        features = self.melspec(waveform)
        if self.amp_to_db is not None:
            features = self.amp_to_db(features)

        features = features.squeeze(0).transpose(0, 1)  # [time, mel]
        return features


def _load_global_cmvn(stats_path: Optional[Path]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if stats_path is None:
        return None

    stats_path = Path(stats_path)
    if not stats_path.is_file():
        raise FileNotFoundError(f"Global CMVN stats not found: {stats_path}")

    stats = np.load(stats_path)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    if "var" in stats:
        std = torch.tensor(stats["var"], dtype=torch.float32).sqrt()
    elif "std" in stats:
        std = torch.tensor(stats["std"], dtype=torch.float32)
    else:
        raise KeyError(f"Global CMVN stats must contain 'var' or 'std': {stats_path}")
    return mean, std


class S2STManifestDataset(Dataset):
    """
    Dataset that reads simultaneous speech-to-speech examples from a TSV manifest.
    """

    def __init__(
        self,
        manifest_path: str,
        *,
        data_root: Optional[str] = None,
        units_root: Optional[str] = None,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
        src_vocab_path: Optional[str] = None,
        tgt_vocab_path: Optional[str] = None,
        text_level: str = "word",
        global_cmvn_stats: Optional[str] = None,
        load_waveform: bool = False,
        load_tgt_audio: bool = False,
        load_tgt_units: bool = False,
        streaming_chunk_ms: Optional[float] = None,
        streaming_hop_ms: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        pad_value: float = 0.0,
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        entries, manifest_root = self._load_manifest(self.manifest_path)
        self.data_root = Path(data_root) if data_root else manifest_root
        self.data_root = self.data_root.resolve() if self.data_root is not None else None

        self.units_root = Path(units_root).resolve() if units_root else self.data_root

        self.entries = self._filter_entries(entries, min_duration, max_duration)

        self.src_tokenizer = TextTokenizer(src_vocab_path, level=text_level)
        self.tgt_tokenizer = TextTokenizer(tgt_vocab_path, level=text_level)
        self.src_tokenizer.update_from_corpus(e.src_text for e in self.entries)
        self.tgt_tokenizer.update_from_corpus(e.tgt_text for e in self.entries)

        self.feature_extractor = SpeechFeatureExtractor(
            sample_rate=sample_rate,
            num_mel_bins=num_mel_bins,
        )
        self.frame_shift = self.feature_extractor.frame_shift()

        cmvn = _load_global_cmvn(global_cmvn_stats)
        self.cmvn_mean = cmvn[0] if cmvn is not None else None
        self.cmvn_std = cmvn[1] if cmvn is not None else None

        self.load_waveform = load_waveform
        self.load_tgt_audio = load_tgt_audio
        self.load_tgt_units = load_tgt_units

        self.streaming_chunk_ms = streaming_chunk_ms
        self.streaming_hop_ms = streaming_hop_ms or streaming_chunk_ms
        self.pad_value = pad_value

        hop_seconds = self.frame_shift
        if self.streaming_chunk_ms:
            self.streaming_chunk_frames = max(1, int(round((self.streaming_chunk_ms / 1000.0) / hop_seconds)))
        else:
            self.streaming_chunk_frames = None

        if self.streaming_hop_ms:
            self.streaming_hop_frames = max(1, int(round((self.streaming_hop_ms / 1000.0) / hop_seconds)))
        else:
            self.streaming_hop_frames = None

    @staticmethod
    def _load_manifest(path: Path) -> Tuple[List[ManifestEntry], Optional[Path]]:
        entries: List[ManifestEntry] = []
        manifest_root: Optional[Path] = None

        with path.open("r", encoding="utf-8") as f:
            first_line = f.readline()
            if not first_line:
                raise ValueError(f"Manifest is empty: {path}")
            first_line = first_line.strip()

            if "\t" in first_line and first_line.split("\t")[0] == "id":
                headers = first_line.split("\t")
            else:
                manifest_root = Path(first_line)
                header_line = f.readline()
                if not header_line:
                    raise ValueError(f"Manifest missing header row: {path}")
                headers = header_line.strip().split("\t")

            reader = csv.DictReader(f, fieldnames=headers, delimiter="\t")

            required_cols = {"id", "src_audio", "src_text", "tgt_text"}
            missing = required_cols.difference(headers)
            if missing:
                raise ValueError(f"Manifest missing required columns: {missing}")

            for row in reader:
                if not row:
                    continue
                sample_id = row.get("id")
                if sample_id is None or sample_id == "id":
                    continue

                src_audio = Path(row["src_audio"])
                tgt_audio_val = row.get("tgt_audio")
                tgt_audio = Path(tgt_audio_val) if tgt_audio_val else None
                tgt_units_val = row.get("tgt_units")
                tgt_units = Path(tgt_units_val) if tgt_units_val else None

                duration_val = row.get("duration") or row.get("audio_len") or row.get("n_frames")
                duration = float(duration_val) if duration_val else None

                entries.append(
                    ManifestEntry(
                        sample_id=sample_id,
                        src_audio=src_audio,
                        src_text=row["src_text"],
                        tgt_audio=tgt_audio,
                        tgt_text=row["tgt_text"],
                        tgt_units=tgt_units,
                        src_lang=row.get("src_lang"),
                        tgt_lang=row.get("tgt_lang"),
                        speaker=row.get("speaker"),
                        duration=duration,
                    )
                )

        return entries, manifest_root

    @staticmethod
    def _filter_entries(
        entries: List[ManifestEntry],
        min_duration: Optional[float],
        max_duration: Optional[float],
    ) -> List[ManifestEntry]:
        if min_duration is None and max_duration is None:
            return entries

        filtered: List[ManifestEntry] = []
        for entry in entries:
            dur = entry.duration
            if dur is None:
                filtered.append(entry)
                continue
            if min_duration is not None and dur < min_duration:
                continue
            if max_duration is not None and dur > max_duration:
                continue
            filtered.append(entry)

        return filtered

    def __len__(self) -> int:
        return len(self.entries)

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute() or self.data_root is None:
            return path
        return self.data_root / path

    def _resolve_units_path(self, path: Path) -> Path:
        if path.is_absolute() or self.units_root is None:
            return path
        return self.units_root / path

    @staticmethod
    def _load_text_units(path: Path) -> np.ndarray:
        units: List[int] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                units.extend(int(tok) for tok in line.split())
        return np.asarray(units, dtype=np.int64)

    def _load_units(self, units_path: Path) -> torch.Tensor:
        resolved = self._resolve_units_path(units_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Unit file not found: {resolved}")

        if resolved.suffix in {".npy", ".npz"}:
            arr = np.load(resolved)
            if resolved.suffix == ".npz":
                # take first array in npz
                key = list(arr.keys())[0]
                arr = arr[key]
        else:
            arr = self._load_text_units(resolved)

        return torch.tensor(arr, dtype=torch.long)

    def _chunk_features(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.streaming_chunk_frames is None:
            raise ValueError("Streaming chunk parameters are not configured.")

        chunk_size = self.streaming_chunk_frames
        hop = self.streaming_hop_frames or chunk_size
        feat_dim = features.size(-1)

        chunks: List[torch.Tensor] = []
        chunk_lengths: List[int] = []

        for start in range(0, features.size(0), hop):
            end = start + chunk_size
            chunk = features[start:end]
            chunk_len = chunk.size(0)

            if chunk_len < chunk_size:
                pad = torch.full(
                    (chunk_size - chunk_len, feat_dim),
                    self.pad_value,
                    dtype=features.dtype,
                    device=features.device,
                )
                chunk = torch.cat([chunk, pad], dim=0)

            chunks.append(chunk)
            chunk_lengths.append(chunk_len)

        if not chunks:
            pad = torch.full(
                (chunk_size, feat_dim),
                self.pad_value,
                dtype=features.dtype,
                device=features.device,
            )
            chunks.append(pad)
            chunk_lengths.append(0)

        return torch.stack(chunks, dim=0), torch.tensor(chunk_lengths, dtype=torch.long)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[index]

        audio_path = self._resolve_path(entry.src_audio)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Source audio not found: {audio_path}")

        waveform, sr = torchaudio.load(audio_path)
        processed_waveform = waveform
        processed_sr = sr
        if sr != self.feature_extractor.sample_rate:
            processed_waveform = resample(waveform, orig_freq=sr, new_freq=self.feature_extractor.sample_rate)
            processed_sr = self.feature_extractor.sample_rate

        if processed_waveform.dim() > 1 and processed_waveform.size(0) > 1:
            processed_waveform = processed_waveform.mean(dim=0, keepdim=True)

        features = self.feature_extractor(processed_waveform, processed_sr)

        if self.cmvn_mean is not None and self.cmvn_std is not None:
            features = (features - self.cmvn_mean) / (self.cmvn_std + 1e-8)

        src_tokens = torch.tensor(
            self.src_tokenizer.encode(entry.src_text, add_bos=False, add_eos=False),
            dtype=torch.long,
        )
        tgt_tokens = torch.tensor(
            self.tgt_tokenizer.encode(entry.tgt_text, add_bos=False, add_eos=True),
            dtype=torch.long,
        )

        duration_sec = processed_waveform.size(-1) / float(self.feature_extractor.sample_rate)

        sample: Dict[str, Any] = {
            "id": entry.sample_id,
            "speech": features,
            "speech_length": torch.tensor(features.size(0), dtype=torch.long),
            "num_frames": torch.tensor(features.size(0), dtype=torch.long),
            "duration_sec": torch.tensor(duration_sec, dtype=torch.float32),
            "src_text": entry.src_text,
            "tgt_text": entry.tgt_text,
            "src_tokens": src_tokens,
            "src_length": torch.tensor(src_tokens.size(0), dtype=torch.long),
            "src_pad_id": torch.tensor(self.src_tokenizer.pad_id, dtype=torch.long),
            "tgt_tokens": tgt_tokens,
            "tgt_length": torch.tensor(tgt_tokens.size(0), dtype=torch.long),
            "tgt_pad_id": torch.tensor(self.tgt_tokenizer.pad_id, dtype=torch.long),
            "tgt_bos_id": torch.tensor(self.tgt_tokenizer.bos_id, dtype=torch.long),
            "src_lang": entry.src_lang,
            "tgt_lang": entry.tgt_lang,
            "speaker": entry.speaker,
        }

        if self.load_waveform:
            sample["waveform"] = processed_waveform
            sample["waveform_sample_rate"] = torch.tensor(self.feature_extractor.sample_rate, dtype=torch.long)
            sample["waveform_length"] = torch.tensor(processed_waveform.size(-1), dtype=torch.long)

        if self.load_tgt_audio and entry.tgt_audio:
            tgt_audio_path = self._resolve_path(entry.tgt_audio)
            if not tgt_audio_path.is_file():
                raise FileNotFoundError(f"Target audio not found: {tgt_audio_path}")
            tgt_waveform, tgt_sr = torchaudio.load(tgt_audio_path)
            if tgt_sr != self.feature_extractor.sample_rate:
                tgt_waveform = resample(tgt_waveform, orig_freq=tgt_sr, new_freq=self.feature_extractor.sample_rate)
                tgt_sr = self.feature_extractor.sample_rate
            sample["tgt_waveform"] = tgt_waveform
            sample["tgt_waveform_length"] = torch.tensor(tgt_waveform.size(-1), dtype=torch.long)
            sample["tgt_waveform_sample_rate"] = torch.tensor(tgt_sr, dtype=torch.long)

        if self.load_tgt_units and entry.tgt_units:
            tgt_unit_tensor = self._load_units(entry.tgt_units)
            sample["tgt_units"] = tgt_unit_tensor
            sample["tgt_unit_length"] = torch.tensor(tgt_unit_tensor.size(0), dtype=torch.long)

        if self.streaming_chunk_frames is not None:
            chunks, chunk_lengths = self._chunk_features(features)
            sample["speech_chunks"] = chunks
            sample["speech_chunk_lengths"] = chunk_lengths

        return sample


def collate_s2st_batches(
    batch: List[Dict[str, torch.Tensor]],
    *,
    pad_value: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for S2ST batches.
    """

    if len(batch) == 0:
        raise ValueError("Batch is empty")

    # Sort by speech length (descending) for better packing
    batch.sort(key=lambda x: x["speech_length"], reverse=True)

    tgt_pad_id = int(batch[0]["tgt_pad_id"])
    tgt_bos_id = int(batch[0]["tgt_bos_id"])
    src_pad_id = int(batch[0]["src_pad_id"])

    # Determine max lengths
    max_speech_len = int(max(item["speech_length"] for item in batch))
    feat_dim = batch[0]["speech"].size(-1)
    max_src_len = int(max(item["src_length"] for item in batch))
    max_tgt_len = int(max(item["tgt_length"] for item in batch))

    speech_batch = torch.full(
        (len(batch), max_speech_len, feat_dim),
        pad_value,
        dtype=batch[0]["speech"].dtype,
    )
    speech_lengths = torch.zeros(len(batch), dtype=torch.long)

    src_batch = torch.full(
        (len(batch), max_src_len),
        fill_value=src_pad_id,
        dtype=torch.long,
    )
    src_lengths = torch.zeros(len(batch), dtype=torch.long)

    tgt_batch = torch.full(
        (len(batch), max_tgt_len),
        fill_value=tgt_pad_id,
        dtype=torch.long,
    )
    prev_output_batch = torch.full(
        (len(batch), max_tgt_len),
        fill_value=tgt_pad_id,
        dtype=torch.long,
    )
    tgt_lengths = torch.zeros(len(batch), dtype=torch.long)

    ids: List[str] = []
    src_langs: List[Optional[str]] = []
    tgt_langs: List[Optional[str]] = []
    speakers: List[Optional[str]] = []
    durations = []
    num_frames = []
    waveforms = []
    wave_lengths = []
    tgt_waveforms = []
    tgt_wave_lengths = []
    chunk_tensors: Optional[torch.Tensor] = None
    chunk_lengths_batch: Optional[torch.Tensor] = None
    has_chunks = "speech_chunks" in batch[0]
    has_units = "tgt_units" in batch[0]

    if has_chunks:
        chunk_time = batch[0]["speech_chunks"].size(1)
        chunk_feat = batch[0]["speech_chunks"].size(2)
        max_chunks = max(item["speech_chunks"].size(0) for item in batch)
        chunk_tensors = torch.full(
            (len(batch), max_chunks, chunk_time, chunk_feat),
            pad_value,
            dtype=batch[0]["speech_chunks"].dtype,
        )
        chunk_lengths_batch = torch.zeros((len(batch), max_chunks), dtype=torch.long)

    if has_units:
        max_unit_len = max(int(item["tgt_unit_length"]) for item in batch)
        unit_batch = torch.full((len(batch), max_unit_len), fill_value=-1, dtype=torch.long)
        unit_lengths = torch.zeros(len(batch), dtype=torch.long)
    else:
        unit_batch = None
        unit_lengths = None

    for idx, item in enumerate(batch):
        speech = item["speech"]
        slen = int(item["speech_length"])
        speech_batch[idx, :slen] = speech
        speech_lengths[idx] = slen

        src_tokens = item["src_tokens"]
        src_len = int(item["src_length"])
        src_batch[idx, :src_len] = src_tokens
        src_lengths[idx] = src_len

        tgt_tokens = item["tgt_tokens"]
        tgt_len = int(item["tgt_length"])
        tgt_batch[idx, :tgt_len] = tgt_tokens

        prev_output_batch[idx, 0] = tgt_bos_id
        if tgt_len > 1:
            prev_output_batch[idx, 1:tgt_len] = tgt_tokens[:-1]
        tgt_lengths[idx] = tgt_len

        ids.append(item["id"])
        src_langs.append(item.get("src_lang"))
        tgt_langs.append(item.get("tgt_lang"))
        speakers.append(item.get("speaker"))
        durations.append(item["duration_sec"])
        num_frames.append(item["num_frames"])

        if "waveform" in item:
            waveforms.append(item["waveform"])
            wave_lengths.append(item["waveform_length"])
        if "tgt_waveform" in item:
            tgt_waveforms.append(item["tgt_waveform"])
            tgt_wave_lengths.append(item["tgt_waveform_length"])

        if has_chunks and chunk_tensors is not None and chunk_lengths_batch is not None:
            chunks = item["speech_chunks"]
            chunk_len = chunks.size(0)
            chunk_tensors[idx, :chunk_len] = chunks
            chunk_lengths_batch[idx, :chunk_len] = item["speech_chunk_lengths"]

        if has_units and unit_batch is not None and unit_lengths is not None:
            ulength = int(item["tgt_unit_length"])
            unit_batch[idx, :ulength] = item["tgt_units"][:ulength]
            unit_lengths[idx] = ulength

    batch_dict: Dict[str, Any] = {
        "id": ids,
        "speech": speech_batch,
        "speech_lengths": speech_lengths,
        "src_tokens": src_batch,
        "src_lengths": src_lengths,
        "target_text": tgt_batch,
        "target_lengths": tgt_lengths,
        "prev_output_tokens": prev_output_batch,
        "duration_sec": torch.stack(durations),
        "num_frames": torch.stack(num_frames),
        "src_lang": src_langs,
        "tgt_lang": tgt_langs,
        "speaker": speakers,
    }

    if waveforms:
        batch_dict["waveform"] = waveforms
        batch_dict["waveform_length"] = torch.stack(wave_lengths)
        batch_dict["waveform_sample_rate"] = batch[0]["waveform_sample_rate"]

    if tgt_waveforms:
        batch_dict["tgt_waveform"] = tgt_waveforms
        batch_dict["tgt_waveform_length"] = torch.stack(tgt_wave_lengths)
        batch_dict["tgt_waveform_sample_rate"] = batch[0]["tgt_waveform_sample_rate"]

    if has_chunks and chunk_tensors is not None and chunk_lengths_batch is not None:
        batch_dict["speech_chunks"] = chunk_tensors
        batch_dict["speech_chunk_lengths"] = chunk_lengths_batch

    if has_units and unit_batch is not None and unit_lengths is not None:
        batch_dict["tgt_units"] = unit_batch
        batch_dict["tgt_unit_lengths"] = unit_lengths

    return batch_dict

