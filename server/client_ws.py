"""
Real-time microphone client for EchoStream FastAPI WebSocket endpoint.

Usage:
    python server/client_ws.py --host 127.0.0.1 --port 8000
Press Ctrl+C to stop recording; the translated audio will be played in real-time through speakers.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import wave
import sys
import io
import queue

import numpy as np
import sounddevice as sd
import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream microphone audio to EchoStream server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--chunk", type=int, default=1024, help="Microphone chunk size in frames")
    parser.add_argument("--outfile", type=str, default="", help="Output WAV filename (optional, if not provided, audio will be played in real-time)")
    return parser.parse_args()


async def stream_microphone(host: str, port: int, chunk: int, outfile: str):
    url = f"ws://{host}:{port}/ws"
    sample_rate = 16000
    channels = 1

    print(f"Connecting to {url}")

    async with websockets.connect(url) as websocket:
        print("🎙️  Speak now (Ctrl+C to finish)...")
        stop_recording = asyncio.Event()
        loop = asyncio.get_event_loop()
        
        def signal_handler(signum, frame):
            """Signal handler that sets the stop event."""
            loop.call_soon_threadsafe(stop_recording.set)
        
        # Signal handler 등록
        signal.signal(signal.SIGINT, signal_handler)
        
        async def send_audio():
            try:
                with sd.InputStream(samplerate=sample_rate, channels=channels, dtype="int16", blocksize=chunk) as stream:
                    while not stop_recording.is_set():
                        try:
                            audio_chunk, _ = stream.read(chunk)
                            await websocket.send(audio_chunk.tobytes())
                        except Exception as e:
                            if not stop_recording.is_set():
                                print(f"Error sending audio: {e}")
                            break
            except Exception as e:
                if not stop_recording.is_set():
                    print(f"Error in audio stream: {e}")
                stop_recording.set()
        
        async def receive_and_play_translations(websocket):
            """서버에서 실시간으로 번역된 오디오를 받아서 바로 재생"""
            try:
                import queue
                audio_queue = queue.Queue()
                
                async def receive_audio():
                    """오디오 데이터를 받아서 큐에 넣기"""
                    while True:
                        try:
                            message = await websocket.recv()
                            if isinstance(message, bytes):
                                # Raw PCM 데이터 (int16)를 float32로 변환
                                audio_data = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                                audio_queue.put(audio_data)
                            elif isinstance(message, dict) and "error" in message:
                                print(f"\n❌ Server error: {message['error']}")
                                audio_queue.put(None)  # 종료 신호
                                break
                        except websockets.exceptions.ConnectionClosed:
                            audio_queue.put(None)  # 종료 신호
                            break
                        except Exception as e:
                            print(f"\n❌ Error receiving translation: {e}")
                            audio_queue.put(None)  # 종료 신호
                            break
                
                # 오디오 수신 태스크 시작
                receive_task = asyncio.create_task(receive_audio())
                
                # 오디오 재생
                with sd.OutputStream(samplerate=sample_rate, channels=channels, dtype="float32", blocksize=1024) as output_stream:
                    while True:
                        try:
                            # 큐에서 오디오 데이터 가져오기 (타임아웃 설정)
                            audio_data = audio_queue.get(timeout=0.1)
                            if audio_data is None:  # 종료 신호
                                break
                            
                            # 실시간 재생
                            output_stream.write(audio_data)
                            print("🔊", end="", flush=True)
                        except queue.Empty:
                            # 큐가 비어있으면 계속 대기
                            await asyncio.sleep(0.01)
                            continue
                        except Exception as e:
                            print(f"\n❌ Error playing audio: {e}")
                            break
                
                # 수신 태스크 종료 대기
                try:
                    await receive_task
                except:
                    pass
                    
            except Exception as e:
                print(f"\n❌ Error in receive_and_play_translations: {e}")
        
        # 오디오 전송 및 수신을 동시에 처리
        send_task = asyncio.create_task(send_audio())
        receive_task = asyncio.create_task(receive_and_play_translations(websocket))
        
        try:
            # stop_recording이 설정될 때까지 대기
            await stop_recording.wait()
            print("\n⏳ Finishing translation...")
        except Exception as e:
            print(f"Error: {e}")
            stop_recording.set()
        
        # 태스크가 완료될 때까지 잠시 대기
        try:
            await asyncio.wait_for(send_task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        # END 신호 전송
        try:
            await websocket.send("END")
        except Exception as e:
            print(f"❌ Error sending END: {e}")
        
        # 마지막 번역 결과 수신 대기
        try:
            await asyncio.wait_for(receive_task, timeout=10.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        
        print(f"\n✅ Real-time translation completed")


def main():
    args = parse_args()
    try:
        asyncio.run(stream_microphone(args.host, args.port, args.chunk, args.outfile))
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()
