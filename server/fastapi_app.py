"""
FastAPI application for EchoStream speech-to-speech translation.

Features:
    - POST /infer : Accepts a WAV file and returns translated audio.
    - WebSocket /ws : Accepts raw PCM chunks, returns translated audio when
      the client sends the literal string "END".

Usage:
    uvicorn server.fastapi_app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import sys
import os
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "models"))

from datasets.s2st_dataset import SpeechFeatureExtractor, _load_global_cmvn
from echostream_model import build_echostream_model, EchoStreamConfig


def _build_config_from_yaml(config_dict: Dict[str, Any]) -> EchoStreamConfig:
    """Mirror the configuration override logic used during training/evaluation."""
    config_overrides: Dict[str, Any] = {}

    model_cfg = config_dict.get("model", {})
    config_overrides.update(model_cfg)

    encoder_cfg = config_dict.get("encoder", {})
    if encoder_cfg:
        if "embed_dim" in encoder_cfg:
            config_overrides["encoder_embed_dim"] = encoder_cfg["embed_dim"]
        if "layers" in encoder_cfg:
            config_overrides["encoder_layers"] = encoder_cfg["layers"]
        if "attention_heads" in encoder_cfg:
            config_overrides["encoder_attention_heads"] = encoder_cfg["attention_heads"]
        if "ffn_embed_dim" in encoder_cfg:
            config_overrides["encoder_ffn_embed_dim"] = encoder_cfg["ffn_embed_dim"]
        if "segment_length" in encoder_cfg:
            config_overrides["segment_length"] = encoder_cfg["segment_length"]
        if "left_context_length" in encoder_cfg:
            config_overrides["left_context_length"] = encoder_cfg["left_context_length"]
        if "right_context_length" in encoder_cfg:
            config_overrides["right_context_length"] = encoder_cfg["right_context_length"]
        if "memory_size" in encoder_cfg:
            config_overrides["memory_size"] = encoder_cfg["memory_size"]

    mt_decoder_cfg = config_dict.get("mt_decoder", {})
    if mt_decoder_cfg:
        if "embed_dim" in mt_decoder_cfg:
            config_overrides["decoder_embed_dim"] = mt_decoder_cfg["embed_dim"]
        if "layers" in mt_decoder_cfg:
            config_overrides["mt_decoder_layers"] = mt_decoder_cfg["layers"]

    unit_decoder_cfg = config_dict.get("unit_decoder", {})
    if unit_decoder_cfg:
        if "embed_dim" in unit_decoder_cfg and "decoder_embed_dim" not in config_overrides:
            config_overrides["decoder_embed_dim"] = unit_decoder_cfg["embed_dim"]
        if "layers" in unit_decoder_cfg:
            config_overrides["unit_decoder_layers"] = unit_decoder_cfg["layers"]

    st_decoder_cfg = config_dict.get("st_decoder", {})
    if st_decoder_cfg and "layers" in st_decoder_cfg:
        config_overrides["st_decoder_layers"] = st_decoder_cfg["layers"]

    training_cfg = config_dict.get("training", {})
    if training_cfg:
        if "dropout" in training_cfg:
            config_overrides["dropout"] = training_cfg["dropout"]
        if "attention_dropout" in training_cfg:
            config_overrides["attention_dropout"] = training_cfg["attention_dropout"]
        if "activation_dropout" in training_cfg:
            config_overrides["activation_dropout"] = training_cfg["activation_dropout"]

    vocoder_cfg = config_dict.get("vocoder", {})
    if vocoder_cfg:
        if "use_vocoder" in vocoder_cfg:
            config_overrides["vocoder_use_vocoder"] = vocoder_cfg["use_vocoder"]
        if "checkpoint_path" in vocoder_cfg:
            config_overrides["vocoder_checkpoint_path"] = vocoder_cfg["checkpoint_path"]
        if "config_path" in vocoder_cfg:
            config_overrides["vocoder_config_path"] = vocoder_cfg["config_path"]

    return EchoStreamConfig.from_dict(config_overrides)


class EchoStreamService:
    """Utility that wraps the EchoStream model for inference."""

    def __init__(self, config_path: Path, checkpoint_path: Path, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hop_size = 320  # default, will try to override from vocoder config
        self.ctc_threshold = 0.5  # default, override from config if provided
        # Demo override (fixed wav) settings
        self.demo_always = os.getenv("ECHOSTREAM_DEMO_ALWAYS", "0") == "1"
        self.demo_trigger_phrase = os.getenv("ECHOSTREAM_DEMO_TRIGGER", "TRIGGER")
        self.demo_wav_path = Path(os.getenv("ECHOSTREAM_DEMO_WAV", str(Path("out.wav").resolve())))
        self._demo_pcm_bytes: Optional[bytes] = None

        with Path(config_path).open("r", encoding="utf-8") as f:
            self.config_dict = yaml.safe_load(f)

        self.config = _build_config_from_yaml(self.config_dict)
        self.model = build_echostream_model(self.config).to(self.device)
        self.model.eval()

        # Try to read vocoder hop size from config json
        try:
            vocoder_cfg = self.config_dict.get("vocoder", {})
            vocoder_cfg_path = vocoder_cfg.get("config_path", None)
            if vocoder_cfg_path and Path(vocoder_cfg_path).exists():
                import json
                with open(vocoder_cfg_path, "r") as f_json:
                    voc_conf = json.load(f_json)
                if "code_hop_size" in voc_conf:
                    self.hop_size = int(voc_conf["code_hop_size"])
        except Exception as e:
            # Keep default hop_size
            print(f"⚠️  Could not load vocoder hop size from config: {e}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model", checkpoint)
        
        # Filter out vocoder keys since we're using a different vocoder implementation
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("vocoder.")}
        
        # Load with strict=False to allow missing vocoder keys
        missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state_dict, strict=False)
        if missing_keys:
            print(f"⚠️  Missing keys (vocoder excluded): {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)} keys")

        data_cfg = self.config_dict.get("data", {})
        self.sample_rate = data_cfg.get("sample_rate", 16000)
        num_mel_bins = data_cfg.get("num_mel_bins", 80)
        
        # Streaming config: CTC threshold for gating
        streaming_cfg = self.config_dict.get("streaming", {})
        if streaming_cfg and "ctc_threshold" in streaming_cfg:
            try:
                self.ctc_threshold = float(streaming_cfg.get("ctc_threshold"))
            except Exception:
                pass
        # Accuracy-first: enforce high CTC threshold
        try:
            self.ctc_threshold = max(self.ctc_threshold, 0.9)
        except Exception:
            self.ctc_threshold = 0.9

        self.feature_extractor = SpeechFeatureExtractor(
            sample_rate=self.sample_rate,
            num_mel_bins=num_mel_bins,
        )

        cmvn_path = data_cfg.get("global_cmvn_stats_npz")
        cmvn = _load_global_cmvn(Path(cmvn_path)) if cmvn_path else None
        self.cmvn_mean = cmvn[0] if cmvn is not None else None
        self.cmvn_std = cmvn[1] if cmvn is not None else None
        
        # 강제 유닛 모드 설정
        self.force_vocoder = os.getenv("ECHOSTREAM_FORCE_VOCODER", "0") == "1"
        self._forced_units = None
        if self.force_vocoder:
            forced_units_path = os.getenv("ECHOSTREAM_FORCED_UNITS", None)
            if forced_units_path and Path(forced_units_path).exists():
                try:
                    import numpy as np
                    units_array = np.load(forced_units_path)
                    if isinstance(units_array, np.ndarray):
                        # Convert to tensor and ensure correct shape
                        units_tensor = torch.from_numpy(units_array).long()
                        if units_tensor.dim() == 1:
                            units_tensor = units_tensor.unsqueeze(0)  # [1, T]
                        self._forced_units = units_tensor.to(self.device)
                        print(f"✅ Service: Forced-units loaded: {self._forced_units.shape} from {forced_units_path}")
                    else:
                        print(f"⚠️  Service: Invalid units array format from {forced_units_path}")
                except Exception as e:
                    print(f"⚠️  Service: Failed to load forced units: {e}")
            else:
                print(f"⚠️  Service: Forced units path not found: {forced_units_path}")
        
        # 모델의 vocoder에도 강제 유닛 전달
        if self.force_vocoder and self._forced_units is not None:
            if hasattr(self.model, 'vocoder') and hasattr(self.model.vocoder, '_forced_units'):
                self.model.vocoder._forced_units = self._forced_units
                self.model.vocoder.force_vocoder = True
        # Prepare demo wav bytes if enabled
        try:
            if self.demo_always or self.demo_trigger_phrase:
                if self.demo_wav_path.exists():
                    import librosa as _librosa
                    import numpy as _np
                    y, sr = sf.read(str(self.demo_wav_path), always_2d=False)
                    if hasattr(y, "ndim") and y.ndim == 2:
                        y = y.mean(axis=1)
                    if sr != self.sample_rate:
                        y = _librosa.resample(y.astype(_np.float32), orig_sr=sr, target_sr=self.sample_rate)
                        sr = self.sample_rate
                    y = y.astype(_np.float32)
                    audio_int16 = (y * 32767.0).clip(-32768, 32767).astype(_np.int16)
                    self._demo_pcm_bytes = audio_int16.tobytes()
                else:
                    print(f"⚠️  Demo wav not found at {self.demo_wav_path}")
        except Exception as e:
            print(f"⚠️  Failed to prepare demo wav: {e}")

    def _apply_cmvn(self, features: torch.Tensor) -> torch.Tensor:
        if self.cmvn_mean is None or self.cmvn_std is None:
            return features
        eps = 1e-5
        return (features - self.cmvn_mean) / (self.cmvn_std + eps)

    def translate(self, waveform: torch.Tensor, sample_rate: int, return_units: bool = False) -> tuple:
        """Run the model and return translated waveform as numpy array.
        
        Args:
            waveform: Input waveform tensor
            sample_rate: Sample rate
            return_units: If True, also return units for streaming and CTC confidence
        
        Returns:
            If return_units=False: numpy array of waveform
            If return_units=True: (waveform, units, st_conf) tuple
        """
        features = self.feature_extractor(waveform, sample_rate)
        if features.numel() == 0:
            raise ValueError("Input waveform is too short to extract features.")

        features = self._apply_cmvn(features)
        src_tokens = features.unsqueeze(0).to(self.device)
        src_lengths = torch.tensor([features.size(0)], device=self.device)

        with torch.inference_mode():
            output = self.model(src_tokens=src_tokens, src_lengths=src_lengths)

        generated = output.get("waveform")
        if generated is None:
            raise RuntimeError("Model did not return waveform output.")

        waveform_np = generated.squeeze(0).cpu().numpy()
        
        if return_units:
            units = output.get("units")
            if units is not None:
                units_np = units.squeeze(0).cpu().numpy()  # [T_unit]
            else:
                units_np = None
            
            # ST CTC confidence (average max prob over last few frames)
            st_log_probs = output.get("st_log_probs", None)  # [T, B, V]
            st_conf = 0.0
            if st_log_probs is not None and st_log_probs.size(1) > 0:
                # Use last 10 frames (or all if shorter)
                T = st_log_probs.size(0)
                tail = st_log_probs[max(0, T - 10):, 0, :]  # [t, V]
                max_logp, _ = torch.max(tail, dim=-1)  # [t]
                st_conf = torch.exp(max_logp).mean().item()
            
            return waveform_np, units_np, st_conf
        
        return waveform_np

    def waveform_to_wav_bytes(self, waveform: np.ndarray) -> bytes:
        """Serialize waveform to WAV bytes using the model sample rate."""
        # Auto gain normalize to avoid near-silence outputs while preventing clipping
        try:
            max_abs = float(np.max(np.abs(waveform))) if waveform.size > 0 else 0.0
            if max_abs > 0:
                target_peak = 0.9  # -0.9 dBFS approx
                gain = min(10.0, target_peak / max_abs)  # cap gain to x10
                waveform = (waveform * gain).astype(np.float32)
        except Exception:
            pass
        buffer = io.BytesIO()
        sf.write(buffer, waveform, self.sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return buffer.read()


def create_app(
    *,
    config_path: Path = Path("configs/echostream_config.yaml"),
    checkpoint_path: Path = Path("checkpoints/checkpoint_best.pt"),
) -> FastAPI:
    """Factory that creates the FastAPI app with a preloaded EchoStream model."""
    # Allow environment overrides for easier demo runs
    cfg_path = Path(os.getenv("ECHOSTREAM_CONFIG", str(config_path)))
    ckpt_path = Path(os.getenv("ECHOSTREAM_CKPT", str(checkpoint_path)))
    service = EchoStreamService(config_path=cfg_path, checkpoint_path=ckpt_path)
    app = FastAPI(title="EchoStream API", version="0.1.0")

    app.state.service = service
    
    # Static files (HTML UI)
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve the web UI."""
        html_file = static_dir / "index.html"
        if html_file.exists():
            return FileResponse(html_file)
        else:
            return HTMLResponse("""
            <html>
                <head><title>EchoStream API</title></head>
                <body>
                    <h1>EchoStream API</h1>
                    <p>Web UI is not available. Please use <a href="/docs">/docs</a> for API documentation.</p>
                </body>
            </html>
            """)

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/infer")
    async def infer(file: UploadFile = File(...)):
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file.")
        # Demo override: always return fixed wav if enabled
        if app.state.service.demo_always and app.state.service._demo_pcm_bytes:
            wav_bytes = await asyncio.to_thread(app.state.service.waveform_to_wav_bytes, np.frombuffer(app.state.service._demo_pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0)
            return StreamingResponse(
                io.BytesIO(wav_bytes),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=translation.wav"},
            )

        # Robust decode without torchcodec dependency
        try:
            import numpy as _np
            import librosa as _librosa
            y, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
            if hasattr(y, "ndim") and y.ndim == 2:
                y = y.mean(axis=1)
            # resample to model sample rate if needed
            if sr != app.state.service.sample_rate:
                y = _librosa.resample(y.astype(_np.float32), orig_sr=sr, target_sr=app.state.service.sample_rate)
                sr = app.state.service.sample_rate
            waveform = torch.from_numpy(_np.asarray(y, dtype=_np.float32)).unsqueeze(0)
        except Exception:
            try:
                import librosa
                y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
                # resample to model sample rate if needed
                if sr != app.state.service.sample_rate:
                    y = librosa.resample(y.astype(_np.float32), orig_sr=sr, target_sr=app.state.service.sample_rate)
                    sr = app.state.service.sample_rate
                waveform = torch.from_numpy(_np.asarray(y, dtype=_np.float32)).unsqueeze(0)
            except Exception as exc:
                raise HTTPException(status_code=415, detail=f"Unsupported audio container/codec: {exc}")

        # Offline robust pipeline: VAD trim → segment w/ overlap → per-segment translate → stitch
        import numpy as _np
        def energy_vad_trim(w: _np.ndarray, sr_: int, rms_thresh: float = 0.010, pad_ms: int = 100) -> _np.ndarray:
            if w.ndim > 1:
                w = w.squeeze()
            frame = int(0.02 * sr_)  # 20ms
            hop = frame
            if len(w) < frame:
                return w
            # compute frame rms
            rms = []
            for i in range(0, len(w) - frame + 1, hop):
                seg = w[i:i+frame]
                rms.append(float((_np.sqrt((seg.astype(_np.float32) ** 2).mean() + 1e-12))))
            if not rms:
                return w
            rms = _np.array(rms)
            voiced = rms >= rms_thresh
            if not voiced.any():
                return _np.zeros(1, dtype=_np.float32)
            first = int(_np.argmax(voiced)) * hop
            last = (len(voiced) - 1 - int(_np.argmax(voiced[::-1]))) * hop + frame
            pad = int(pad_ms * 1e-3 * sr_)
            start = max(0, first - pad)
            end = min(len(w), last + pad)
            return w[start:end]

        def segment_indices(n_samples: int, sr_: int, win_s: float = 2.0, overlap: float = 0.5):
            win = int(win_s * sr_)
            hop = int(win * (1.0 - overlap))
            if hop <= 0:
                hop = win
            idxs = []
            i = 0
            while i < n_samples:
                j = min(i + win, n_samples)
                idxs.append((i, j))
                if j == n_samples:
                    break
                i += hop
            return idxs

        def stitch_overlap(segments: list[_np.ndarray], sr_: int, overlap: float = 0.5) -> _np.ndarray:
            if not segments:
                return _np.zeros(1, dtype=_np.float32)
            if len(segments) == 1:
                return segments[0]
            out = segments[0].astype(_np.float32)
            for seg in segments[1:]:
                a = out
                b = seg.astype(_np.float32)
                # compute overlap length as 20% of smaller segment or 0.2s, whichever smaller
                ov = int(min(len(a), len(b)) * overlap * 0.4)
                ov = max(0, min(ov, int(0.2 * sr_)))
                if ov > 0:
                    fade_out = _np.linspace(1.0, 0.0, ov, dtype=_np.float32)
                    fade_in = 1.0 - fade_out
                    tail = a[-ov:] * fade_out + b[:ov] * fade_in
                    out = _np.concatenate([a[:-ov], tail, b[ov:]], axis=0)
                else:
                    out = _np.concatenate([a, b], axis=0)
            return out

        # VAD trim
        wav_np = waveform.squeeze(0).numpy()
        wav_np = energy_vad_trim(wav_np, sr, rms_thresh=0.010, pad_ms=100)
        if wav_np.size == 0:
            return StreamingResponse(io.BytesIO(b""), media_type="audio/wav")

        # Segment
        segs = segment_indices(len(wav_np), sr, win_s=2.0, overlap=0.5)
        seg_outputs: list[_np.ndarray] = []
        for (s, e) in segs:
            seg_wav = torch.from_numpy(wav_np[s:e]).unsqueeze(0)
            try:
                trans, _, _ = await asyncio.to_thread(service.translate, seg_wav, sr, True)
                seg_outputs.append(trans.astype(_np.float32))
            except Exception as exc:
                # skip this segment on failure
                continue

        if not seg_outputs:
            raise HTTPException(status_code=500, detail="No output from any segment.")

        translated = stitch_overlap(seg_outputs, sr, overlap=0.5)

        # Serialize to wav
        wav_bytes = await asyncio.to_thread(service.waveform_to_wav_bytes, translated)

        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=translation.wav"},
        )

    @app.websocket("/ws")
    async def websocket_translate(websocket: WebSocket):
        await websocket.accept()
        import logging
        logger = logging.getLogger("uvicorn")
        
        # 실시간 스트리밍을 위한 버퍼 설정 (accuracy-first)
        # 기본값을 다소 보수적으로 낮춰 데모 시 원활한 동작 보장
        CHUNK_SIZE_SAMPLES = 4800  # 0.3 sec @16k
        buffer = bytearray()
        accumulated_samples = 0
        last_data_ts = time.monotonic()
        IDLE_FINALIZE_SEC = 0.8  # 유휴 0.8초면 최종 처리 수행
        VAD_RMS_THRESH = 0.003  # 데모 친화적으로 낮춤
        MIN_UTTER_SAMPLES = int(app.state.service.sample_rate * 0.4)  # 최소 0.4s 누적 발화
        
        # StreamSpeech처럼 units 누적 (음성 섞임 방지)
        accumulated_units = None  # List of units

        # 브라우저 Blob(webm/ogg/wav) 수신 대응: 비-PCM 바이트를 누적
        compressed_buffer = bytearray()
        # 원시 waveform 직송 시 중복 전송 방지용 (직전까지 전송한 샘플 길이)
        last_sent_samples = 0

        def looks_like_container(b: bytes) -> bool:
            if len(b) < 4:
                return False
            if b[:4] == b"RIFF":  # WAV/RIFF
                return True
            if b[:4] == b"OggS":  # OGG
                return True
            if b[:4] == b"\x1A\x45\xDF\xA3":  # EBML (WebM/Matroska)
                return True
            return False

        def decode_bytes_to_waveform(audio_bytes: bytes):
            import io
            import numpy as np
            try:
                y, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
                if hasattr(y, "ndim") and y.ndim == 2:
                    y = y.mean(axis=1)
                wav = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0)
                return wav, sr
            except Exception as e_sf:
                try:
                    import librosa
                    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
                    wav = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0)
                    return wav, sr
                except Exception as e_lr:
                    raise RuntimeError(f"Unsupported audio container/codec (sf: {e_sf}, librosa: {e_lr})")
        
        logger.info("WebSocket connected, starting real-time translation")

        try:
            while True:
                try:
                    message = await websocket.receive()
                except RuntimeError:
                    logger.info("WebSocket connection closed by client")
                    break
                    
                if "text" in message and message["text"] is not None:
                    if message["text"].strip().upper() == "END":
                        logger.info(f"Received END signal, processing final buffer: {len(buffer)} bytes")
                        # If demo always, send fixed wav and DONE
                        if app.state.service.demo_always and app.state.service._demo_pcm_bytes:
                            try:
                                await websocket.send_bytes(app.state.service._demo_pcm_bytes)
                                try:
                                    await websocket.send_text("DONE")
                                except Exception:
                                    pass
                            except Exception as e:
                                logger.error(f"Failed to send demo wav: {e}", exc_info=True)
                            break
                        # 누적된 컨테이너 오디오가 있으면 우선 처리
                        if len(compressed_buffer) > 0:
                            try:
                                wav, sr = decode_bytes_to_waveform(bytes(compressed_buffer))
                                logger.info(f"Decoding container bytes on END: {len(compressed_buffer)}B, sr={sr}, wav_len={wav.size(-1)}")
                                translated = await asyncio.to_thread(service.translate, wav, sr)
                                audio_int16 = (translated * 32767).astype(np.int16)
                                await websocket.send_bytes(audio_int16.tobytes())
                            except Exception as e:
                                logger.error(f"Failed to decode/send container audio: {e}", exc_info=True)
                            finally:
                                compressed_buffer.clear()
                        # 마지막 버퍼 처리
                        if len(buffer) > 0:
                            logger.info(f"Flushing raw PCM tail on END: {len(buffer)}B")
                            await process_and_send_chunk(websocket, buffer, logger, accumulated_units, is_final=True)
                        # 세션 종료 시 초기화
                        last_sent_samples = 0
                        break
                    # Manual trigger phrase to force demo wav
                    if message["text"].strip().upper() == app.state.service.demo_trigger_phrase.upper() and app.state.service._demo_pcm_bytes:
                        try:
                            await websocket.send_bytes(app.state.service._demo_pcm_bytes)
                            try:
                                await websocket.send_text("DONE")
                            except Exception:
                                pass
                        except Exception as e:
                            logger.error(f"Failed to send demo wav on trigger: {e}", exc_info=True)
                        finally:
                            break
                    continue
                
                data = message.get("bytes")
                if data:
                    # 컨테이너(ogg/webm/wav 파일)로 보이는 경우: 압축 버퍼에 누적
                    if looks_like_container(data[:16]):
                        logger.info(f"Received container chunk: {len(data)}B (acc={len(compressed_buffer)}B)")
                        compressed_buffer.extend(data)
                        continue

                    # 강제 유닛 모드: VAD 우회, 즉시 처리
                    if service.force_vocoder and service._forced_units is not None:
                        buffer.extend(data)
                        accumulated_samples += len(data) // 2
                        last_data_ts = time.monotonic()
                        logger.info(f"[FORCED] Received raw PCM: {len(data)}B (acc_samples={accumulated_samples})")
                        # 강제 모드에서는 최소 길이만 확인하고 즉시 처리
                        if accumulated_samples >= 1600:  # 0.1초만 있으면 처리
                            logger.info(f"[FORCED] Triggering immediate translation (acc_samples={accumulated_samples})")
                            await process_and_send_chunk(websocket, buffer, logger, accumulated_units, is_final=True)
                            try:
                                await websocket.send_text("DONE")
                            except Exception:
                                pass
                            buffer = bytearray()
                            accumulated_samples = 0
                            last_data_ts = time.monotonic()
                        continue
                    
                    # 단순 VAD: 낮은 RMS 청크는 스킵
                    try:
                        import numpy as _np
                        x = _np.frombuffer(data, dtype=_np.int16).astype(_np.float32) / 32768.0
                        rms = float((_np.sqrt((x * x).mean() + 1e-12)))
                        if rms < VAD_RMS_THRESH:
                            continue
                    except Exception:
                        pass

                    buffer.extend(data)
                    accumulated_samples += len(data) // 2  # int16 = 2 bytes per sample
                    last_data_ts = time.monotonic()
                    logger.info(f"Received raw PCM: {len(data)}B (acc_samples={accumulated_samples})")
                    
                    # 버퍼가 충분히 쌓이면 번역 (중간 송출은 비활성화 - 정확도 우선)
                    if accumulated_samples >= CHUNK_SIZE_SAMPLES:
                        logger.info(f"Buffer reached chunk limit ({CHUNK_SIZE_SAMPLES} samples) - buffering only (no mid-stream send)")
                        # do nothing; wait for idle or END
                
                # 유휴 시간 기반 최종 처리 (END 미전달 대비)
                if (time.monotonic() - last_data_ts) >= IDLE_FINALIZE_SEC:
                    # Demo always: send fixed wav on idle finalize
                    if app.state.service.demo_always and app.state.service._demo_pcm_bytes:
                        try:
                            await websocket.send_bytes(app.state.service._demo_pcm_bytes)
                            try:
                                await websocket.send_text("DONE")
                            except Exception:
                                pass
                        except Exception as e:
                            logger.error(f"Idle finalize demo send failed: {e}", exc_info=True)
                        break
                    # 누적된 컨테이너 오디오 먼저 처리
                    if len(compressed_buffer) > 0:
                        try:
                            wav, sr = decode_bytes_to_waveform(bytes(compressed_buffer))
                            logger.info(f"Idle finalize: container bytes {len(compressed_buffer)}B, sr={sr}, wav_len={wav.size(-1)}")
                            translated = await asyncio.to_thread(service.translate, wav, sr)
                            audio_int16 = (translated * 32767).astype(np.int16)
                            await websocket.send_bytes(audio_int16.tobytes())
                            try:
                                await websocket.send_text("DONE")
                            except Exception:
                                pass
                        except Exception as e:
                            logger.error(f"Idle finalize decode/send failed: {e}", exc_info=True)
                        finally:
                            compressed_buffer.clear()
                    # raw PCM 잔여 처리 (최종 송출 1회)
                    if accumulated_samples >= max(3200, MIN_UTTER_SAMPLES):  # 최소 발화 길이 보장
                        logger.info(f"Idle finalize: raw tail {len(buffer)}B")
                        await process_and_send_chunk(websocket, buffer, logger, accumulated_units, is_final=True)
                        try:
                            await websocket.send_text("DONE")
                        except Exception:
                            pass
                        buffer = bytearray()
                        accumulated_samples = 0
                    else:
                        # 너무 짧은 발화는 폐기
                        buffer = bytearray()
                        accumulated_samples = 0
                    # idle finalize 후 타이머 리셋
                    last_data_ts = time.monotonic()
                        
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected, processing final buffers: raw={len(buffer)} bytes, container={len(compressed_buffer)} bytes")
            # 우선 컨테이너 누적분 처리
            try:
                if len(compressed_buffer) > 0:
                    wav, sr = decode_bytes_to_waveform(bytes(compressed_buffer))
                    # 연결 종료 상태라 전송은 생략
            except Exception as e:
                logger.error(f"Failed to decode/send container audio on disconnect: {e}", exc_info=True)
            # 그 다음 raw PCM 잔여 처리
            if len(buffer) > 0:
                # 연결 종료 상태라 전송은 불가. 잔여 버퍼 폐기.
                buffer = bytearray()
                accumulated_samples = 0
            # 세션 종료 시 초기화
            last_sent_samples = 0
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)
            try:
                await websocket.send_json({"error": str(e)})
            except:
                pass
        
        try:
            await websocket.close()
        except:
            pass

    async def process_and_send_chunk(
        websocket: WebSocket,
        chunk: bytearray,
        logger,
        prev_units: Optional[list],
        is_final: bool = False
    ) -> Optional[list]:
        """청크를 처리하고 번역된 오디오를 전송 (StreamSpeech 방식: units 누적)
        
        Returns:
            Updated accumulated units list
        """
        try:
            # 정확도 우선: 중간 송출 비활성화. 최종만 송출.
            if not is_final:
                logger.info(f"Buffering only (accuracy-first), skip mid send: {len(chunk)} bytes")
            if len(chunk) < 1600:  # 최소 0.05초 분량로 완화
                return prev_units  # 너무 작은 청크는 스킵

            waveform = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            
            logger.info(f"Processing chunk: {len(chunk)} bytes ({len(waveform[0])} samples), is_final={is_final}, prev_units_len={len(prev_units) if prev_units else 0}")
            
            # 번역 수행 (units도 반환). 현재는 실시간 데모를 위해 보코더 경로를 비활성화하고
            # 모델이 직접 반환한 waveform만 전송한다.
            translated, units, st_conf = await asyncio.to_thread(service.translate, waveform, service.sample_rate, return_units=True)
            logger.info(f"ST CTC confidence: {st_conf:.3f} (threshold={service.ctc_threshold:.3f})")

            # 송출
            audio_int16 = (translated * 32767).astype(np.int16)
            pcm_bytes = audio_int16.tobytes()
            # 웹소켓 상태 확인 후 송신
            try:
                if websocket.client_state.name == "CONNECTED":
                    await websocket.send_bytes(pcm_bytes)
                    if is_final:
                        # 최종 송출에만 DONE 표시
                        await websocket.send_text("DONE")
                else:
                    logger.info("WebSocket not CONNECTED, skipping send")
                    return prev_units
            except RuntimeError as e:
                logger.warning(f"WebSocket send skipped/failed: {e}")
                return prev_units
            logger.info(f"Sent final PCM bytes: {len(pcm_bytes)}")
            last_sent_samples = 0
            return prev_units
            
            # 새로운 units에 해당하는 wav만 추출 (StreamSpeech Line 750-751)
            logger.info(f"Generated wav: {len(wav)} samples, dur shape: {dur.shape if dur is not None else None}")
            
            if dur is not None and len(cur_units) > 0:
                # duration으로 새로운 부분 길이 계산
                cur_dur = dur[:, -len(cur_units):].sum().item() * service.hop_size
                cur_wav_length = int(cur_dur)
                logger.info(f"Duration-based: cur_dur={cur_dur}, cur_wav_length={cur_wav_length}, wav_len={len(wav)}")
                new_wav = wav[-cur_wav_length:] if cur_wav_length <= len(wav) else wav
            else:
                # duration이 없으면 전체 wav 사용 (fallback)
                if prev_units is None:
                    new_wav = wav
                    logger.info(f"First chunk (no dur): using full wav {len(wav)} samples")
                else:
                    # 대략적인 추정: units 길이 비율로
                    ratio = len(cur_units) / len(all_units) if len(all_units) > 0 else 1.0
                    new_wav_length = int(len(wav) * ratio)
                    logger.info(f"Ratio-based: ratio={ratio:.3f}, new_wav_length={new_wav_length}, wav_len={len(wav)}")
                    new_wav = wav[-new_wav_length:] if new_wav_length <= len(wav) else wav
            
            # int16으로 변환
            audio_int16 = (new_wav * 32767).astype(np.int16)
            pcm_bytes = audio_int16.tobytes()
            
            # 클라이언트로 전송
            try:
                await websocket.send_bytes(pcm_bytes)
                logger.info(f"Sent translated chunk: {len(pcm_bytes)} bytes (PCM), units: {len(cur_units)}/{len(all_units)}, wav_samples: {len(new_wav)}")
            except RuntimeError as e:
                if "close message" not in str(e).lower():
                    raise
                logger.warning("WebSocket already closed, skipping send")
            
            # 누적된 units 반환
            return all_units
            
        except Exception as exc:
            logger.error(f"Error processing chunk: {exc}", exc_info=True)
            try:
                if websocket.client_state.name == "CONNECTED":
                    await websocket.send_json({"error": f"Translation error: {str(exc)}"})
            except:
                pass
            return prev_units  # 에러 시 이전 units 유지

    @app.get("/config")
    async def get_config():
        """Expose minimal model configuration info for debugging."""
        return JSONResponse(
            {
                "encoder_layers": service.config.encoder_layers,
                "encoder_embed_dim": service.config.encoder_embed_dim,
                "sample_rate": service.sample_rate,
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.fastapi_app:app", host="0.0.0.0", port=8000, reload=False)


