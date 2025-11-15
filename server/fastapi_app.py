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
from fastapi.responses import StreamingResponse, JSONResponse

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

    return EchoStreamConfig.from_dict(config_overrides)


class EchoStreamService:
    """Utility that wraps the EchoStream model for inference."""

    def __init__(self, config_path: Path, checkpoint_path: Path, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with Path(config_path).open("r", encoding="utf-8") as f:
            self.config_dict = yaml.safe_load(f)

        self.config = _build_config_from_yaml(self.config_dict)
        self.model = build_echostream_model(self.config).to(self.device)
        self.model.eval()

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model", checkpoint)
        self.model.load_state_dict(state_dict)

        data_cfg = self.config_dict.get("data", {})
        self.sample_rate = data_cfg.get("sample_rate", 16000)
        num_mel_bins = data_cfg.get("num_mel_bins", 80)

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

    def translate(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """Run the model and return translated waveform as numpy array."""
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

        return generated.squeeze(0).cpu().numpy()

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
        buffer = bytearray()

        try:
            while True:
                message = await websocket.receive()
                if "text" in message and message["text"] is not None:
                    if message["text"].strip().upper() == "END":
                        break
                    # Optionally allow keeping-alive messages
                    continue
                data = message.get("bytes")
                if data:
                    buffer.extend(data)
        except WebSocketDisconnect:
            return

        if not buffer:
            await websocket.send_json({"error": "No audio received"})
            await websocket.close()
            return

        waveform = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(waveform).unsqueeze(0)

        try:
            translated = await asyncio.to_thread(service.translate, waveform, service.sample_rate)
            wav_bytes = await asyncio.to_thread(service.waveform_to_wav_bytes, translated)
        except Exception as exc:
            await websocket.send_json({"error": str(exc)})
            await websocket.close()
            return

        await websocket.send_bytes(wav_bytes)
        await websocket.close()

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


