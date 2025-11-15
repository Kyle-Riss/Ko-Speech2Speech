"""
Simple microphone client for EchoStream FastAPI WebSocket endpoint.

Usage:
    python server/client_ws.py --host 127.0.0.1 --port 8000
Press Ctrl+C to stop recording; the translated audio will be saved as translation.wav.
"""

from __future__ import annotations

import argparse
import asyncio
import wave

import numpy as np
import sounddevice as sd
import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream microphone audio to EchoStream server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--chunk", type=int, default=1024, help="Microphone chunk size in frames")
    parser.add_argument("--outfile", type=str, default="translation.wav", help="Output WAV filename")
    return parser.parse_args()


async def stream_microphone(host: str, port: int, chunk: int, outfile: str):
    url = f"ws://{host}:{port}/ws"
    sample_rate = 16000
    channels = 1

    print(f"Connecting to {url}")

    async with websockets.connect(url) as websocket:
        print("🎙️  Speak now (Ctrl+C to finish)...")
        try:
            with sd.InputStream(samplerate=sample_rate, channels=channels, dtype="int16", blocksize=chunk) as stream:
                while True:
                    audio_chunk, _ = stream.read(chunk)
                    await websocket.send(audio_chunk.tobytes())
        except KeyboardInterrupt:
            print("\n⏳ Translating...")

        await websocket.send("END")
        translated_bytes = await websocket.recv()

        with wave.open(outfile, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # int16
            wf.setframerate(sample_rate)
            wf.writeframes(translated_bytes)

        print(f"✅ Saved translated audio to {outfile}")


def main():
    args = parse_args()
    asyncio.run(stream_microphone(args.host, args.port, args.chunk, args.outfile))


if __name__ == "__main__":
    main()







