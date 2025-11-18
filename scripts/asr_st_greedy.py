#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import os
import yaml
import torch
import numpy as np
import soundfile as sf
import io

# Local imports
ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

from echostream_model import build_echostream_model, EchoStreamConfig
from datasets.s2st_dataset import SpeechFeatureExtractor, _load_global_cmvn


def load_config_and_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    with config_path.open("r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    # Build model config (mimic server)
    overrides = {}
    enc = cfg_dict.get("encoder", {})
    if enc:
        for k, src in [
            ("encoder_embed_dim", "embed_dim"),
            ("encoder_layers", "layers"),
            ("encoder_attention_heads", "attention_heads"),
            ("encoder_ffn_embed_dim", "ffn_embed_dim"),
            ("segment_length", "segment_length"),
            ("left_context_length", "left_context_length"),
            ("right_context_length", "right_context_length"),
            ("memory_size", "memory_size"),
        ]:
            if src in enc:
                overrides[k] = enc[src]
    mt = cfg_dict.get("mt_decoder", {})
    if mt:
        if "embed_dim" in mt:
            overrides["decoder_embed_dim"] = mt["embed_dim"]
        if "layers" in mt:
            overrides["mt_decoder_layers"] = mt["layers"]
    unitd = cfg_dict.get("unit_decoder", {})
    if unitd:
        if "embed_dim" in unitd and "decoder_embed_dim" not in overrides:
            overrides["decoder_embed_dim"] = unitd["embed_dim"]
        if "layers" in unitd:
            overrides["unit_decoder_layers"] = unitd["layers"]
    st = cfg_dict.get("st_decoder", {})
    if st and "layers" in st:
        overrides["st_decoder_layers"] = st["layers"]

    cfg = EchoStreamConfig.from_dict(overrides)
    model = build_echostream_model(cfg).to(device)
    model.eval()

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location=device)
    sd = state.get("model", state)
    filtered = {k: v for k, v in sd.items() if not k.startswith("vocoder.")}
    model.load_state_dict(filtered, strict=False)

    return model, cfg, cfg_dict


def load_wav(path: Path) -> Tuple[torch.Tensor, int]:
    data, sr = sf.read(str(path), always_2d=False)
    if hasattr(data, "ndim") and data.ndim == 2:
        data = data.mean(axis=1)
    wav = torch.from_numpy(np.asarray(data, dtype=np.float32)).unsqueeze(0)
    return wav, sr


def ctc_greedy_decode(log_probs: torch.Tensor, blank_id: int = 0, pad_id: int = 1) -> List[int]:
    # log_probs: [T, B, V]; assume B=1
    ids = torch.argmax(log_probs, dim=-1)[:, 0].tolist()
    collapsed: List[int] = []
    prev: Optional[int] = None
    for t in ids:
        if t == blank_id or t == pad_id:
            prev = None
            continue
        if prev is None or t != prev:
            collapsed.append(t)
        prev = t
    return collapsed


def load_vocab_tokens(vocab_path: Path) -> List[str]:
    # Simple loader: take token as first entry per line
    tokens: List[str] = []
    with vocab_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tok = line.split()[0]
            tokens.append(tok)
    return tokens


def tokens_to_text(token_ids: List[int], vocab_tokens: List[str]) -> str:
    # Note: Model uses CTC; vocabulary alignment assumes training order.
    # We don't shift indices for special tokens; this yields a rough text.
    words: List[str] = []
    V = len(vocab_tokens)
    for tid in token_ids:
        if 0 <= tid < V:
            words.append(vocab_tokens[tid])
    return " ".join(words)


def main():
    ap = argparse.ArgumentParser(description="Offline ASR/ST greedy decode from a WAV")
    ap.add_argument("--wav", required=True, help="Input WAV (16kHz mono recommended)")
    ap.add_argument("--config", default=str(ROOT / "configs/echostream_config.mini.yaml"))
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/checkpoint_best.pt"))
    ap.add_argument("--print-asr", action="store_true", help="Also print ASR greedy decode")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, cfg_dict = load_config_and_model(Path(args.config), Path(args.checkpoint), device)

    data_cfg = cfg_dict.get("data", {})
    sample_rate = int(data_cfg.get("sample_rate", 16000))
    num_mel_bins = int(data_cfg.get("num_mel_bins", 80))
    global_cmvn = data_cfg.get("global_cmvn_stats_npz")
    tgt_vocab_path = data_cfg.get("tgt_dict")
    src_vocab_path = data_cfg.get("src_dict")

    # Feature extractor and CMVN
    fe = SpeechFeatureExtractor(sample_rate=sample_rate, num_mel_bins=num_mel_bins)
    cmvn = _load_global_cmvn(Path(global_cmvn)) if global_cmvn else None
    cmvn_mean = cmvn[0] if cmvn is not None else None
    cmvn_std = cmvn[1] if cmvn is not None else None

    # Load audio
    wav, sr = load_wav(Path(args.wav))
    # Model's extractor will resample internally via fe.__call__
    feats = fe(wav, sr)
    if feats.numel() == 0:
        print("Input is too short.")
        return
    if cmvn_mean is not None and cmvn_std is not None:
        feats = (feats - cmvn_mean) / (cmvn_std + 1e-5)
    src_tokens = feats.unsqueeze(0).to(device)
    src_lengths = torch.tensor([feats.size(0)], device=device)

    with torch.inference_mode():
        out = model(src_tokens=src_tokens, src_lengths=src_lengths)

    # Decode ST CTC
    st_log_probs = out.get("st_log_probs", None)
    if st_log_probs is not None:
        st_ids = ctc_greedy_decode(st_log_probs.cpu(), blank_id=0, pad_id=1)
        st_text = ""
        if tgt_vocab_path:
            vocab_tokens = load_vocab_tokens(Path(tgt_vocab_path))
            st_text = tokens_to_text(st_ids, vocab_tokens)
        print("ST ids:", st_ids[:64], ("..." if len(st_ids) > 64 else ""))
        print("ST text:", st_text if st_text else "(vocab mapping unavailable)")
    else:
        print("No ST log_probs in output.")

    if args.print_asr:
        asr_log_probs = out.get("asr_log_probs", None)
        if asr_log_probs is not None:
            asr_ids = ctc_greedy_decode(asr_log_probs.cpu(), blank_id=0, pad_id=1)
            asr_text = ""
            if src_vocab_path:
                vocab_tokens = load_vocab_tokens(Path(src_vocab_path))
                asr_text = tokens_to_text(asr_ids, vocab_tokens)
            print("ASR ids:", asr_ids[:64], ("..." if len(asr_ids) > 64 else ""))
            print("ASR text:", asr_text if asr_text else "(vocab mapping unavailable)")
        else:
            print("No ASR log_probs in output.")


if __name__ == "__main__":
    main()



