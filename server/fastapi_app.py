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

        self.feature_extractor = SpeechFeatureExtractor(
            sample_rate=self.sample_rate,
            num_mel_bins=num_mel_bins,
        )

        cmvn_path = data_cfg.get("global_cmvn_stats_npz")
        cmvn = _load_global_cmvn(Path(cmvn_path)) if cmvn_path else None
        self.cmvn_mean = cmvn[0] if cmvn is not None else None
        self.cmvn_std = cmvn[1] if cmvn is not None else None

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
    service = EchoStreamService(config_path=config_path, checkpoint_path=checkpoint_path)
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

        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        waveform = waveform.mean(dim=0, keepdim=True)  # ensure mono

        try:
            translated = await asyncio.to_thread(service.translate, waveform, sr)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

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
        
        # 실시간 스트리밍을 위한 버퍼 설정
        CHUNK_SIZE_SAMPLES = 16000  # 1초 분량 (16kHz)
        buffer = bytearray()
        accumulated_samples = 0
        
        # StreamSpeech처럼 units 누적 (음성 섞임 방지)
        accumulated_units = None  # List of units
        
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
                        # 마지막 버퍼 처리
                        if len(buffer) > 0:
                            await process_and_send_chunk(websocket, buffer, logger, accumulated_units, is_final=True)
                        break
                    continue
                
                data = message.get("bytes")
                if data:
                    buffer.extend(data)
                    accumulated_samples += len(data) // 2  # int16 = 2 bytes per sample
                    
                    # 버퍼가 충분히 쌓이면 번역하고 전송
                    if accumulated_samples >= CHUNK_SIZE_SAMPLES:
                        chunk = buffer[:CHUNK_SIZE_SAMPLES * 2]  # 2 bytes per sample
                        buffer = buffer[CHUNK_SIZE_SAMPLES * 2:]
                        accumulated_samples = 0
                        
                        # 청크를 비동기로 처리하고 전송 (units 누적)
                        accumulated_units = await process_and_send_chunk(websocket, chunk, logger, accumulated_units, is_final=False)
                        
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected, processing final buffer: {len(buffer)} bytes")
            if len(buffer) > 0:
                await process_and_send_chunk(websocket, buffer, logger, accumulated_units, is_final=True)
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
            if len(chunk) < 3200:  # 최소 0.1초 분량
                if not is_final:
                    return prev_units  # 너무 작은 청크는 스킵
            
            waveform = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            
            logger.info(f"Processing chunk: {len(chunk)} bytes ({len(waveform[0])} samples), is_final={is_final}, prev_units_len={len(prev_units) if prev_units else 0}")
            
            # 번역 수행 (units도 반환)
            translated, units, st_conf = await asyncio.to_thread(service.translate, waveform, service.sample_rate, return_units=True)
            logger.info(f"ST CTC confidence: {st_conf:.3f} (threshold={service.ctc_threshold:.3f})")
            
            # CTC confidence gating: if confidence is low and not final, wait for more context
            if not is_final and st_conf < service.ctc_threshold:
                logger.info("CTC confidence below threshold; deferring synthesis for this chunk")
                return prev_units
            
            if units is None:
                # units가 없으면 기존 방식 사용
                audio_int16 = (translated * 32767).astype(np.int16)
                pcm_bytes = audio_int16.tobytes()
                await websocket.send_bytes(pcm_bytes)
                return prev_units
            
            # StreamSpeech 방식: units 누적
            current_units = units.tolist()  # [T_unit]
            logger.info(f"Current units: {len(current_units)} units, prev_units: {len(prev_units) if prev_units else 0} units")
            
            if prev_units is None:
                # 첫 번째 청크: 전체 units 사용
                all_units = current_units
                cur_units = current_units
                logger.info(f"First chunk: all_units={len(all_units)}, cur_units={len(cur_units)}")
            else:
                # 이후 청크: 새로운 units만 추가
                all_units = prev_units + current_units
                cur_units = current_units
                logger.info(f"Subsequent chunk: all_units={len(all_units)}, cur_units={len(cur_units)}")
            
            # 전체 units로 vocoder 호출 (StreamSpeech 방식)
            all_units_tensor = torch.tensor(all_units, dtype=torch.long, device=service.device).unsqueeze(0)  # [1, T_all]
            
            # vocoder 호출 (duration prediction 활성화)
            x = {"code": all_units_tensor, "dur_prediction": True}
            wav, dur = await asyncio.to_thread(lambda: service.model.vocoder.generator(**x))
            wav = wav.detach().cpu().squeeze(0).numpy()  # [T_wav]
            dur = dur.detach().cpu() if dur is not None else None
            
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


