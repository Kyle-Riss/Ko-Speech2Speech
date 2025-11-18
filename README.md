EchoStream - Real-time Speech-to-Speech Translation (Emformer + Units)

This repository contains a FastAPI-based real-time speech-to-speech translation system built around an Emformer encoder, multi-task decoders (ASR/ST/MT/Unit), and a CodeHiFiGAN-based vocoder. It supports low-latency streaming with MT incremental state, CTC gating, and duration-synchronized unit synthesis.

1) What you get

- Real-time server (FastAPI + WebSocket)
- Web client (server/static/index.html) and Python client (server/client_ws.py)
- Streaming-quality improvements: MT incremental state, ST-CTC gating, vocoder duration hop sync
- CPU-safe data loading path (librosa/soundfile fallback when torchaudio/torchcodec is unavailable)
- Mini training config with unit learning enabled (configs/echostream_config.mini.yaml)

2) Requirements

- OS: Linux/macOS (Apple Silicon works; CPU-only supported)
- Python: 3.10–3.12
- PortAudio/libsndfile/ffmpeg (for mic I/O; macOS: brew install portaudio libsndfile ffmpeg)
- Git

3) Clone

Shallow-clone only the working branch (recommended to avoid large history):

```bash
git clone --depth 1 --branch feature/streaming-mini-units https://github.com/Kyle-Riss/EchoStream.git
cd EchoStream
```

4) Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If torchaudio is not usable (torchcodec missing), the project automatically falls back to soundfile/librosa in datasets/s2st_dataset.py.

5) Assets you must provide

We do not commit large assets. Prepare these on your machine (or host them as release artifacts and download via your preferred script).

- pretrain_models/
  - mHuBERT (e.g., pretrain_models/mHuBERT/mhubert_base_vp_en_es_fr_it3.pt)
  - CodeHiFiGAN vocoder (config.json + g_*.pt)
- data/
  - Manifests: train/dev/test tsv (e.g., data/train_sampled.units.tsv, data/dev_sampled.units.tsv, data/test_sampled.tsv)
  - Units: data/units/*.npy
  - gcmvn: data/gcmvn.npz
  - Dictionaries: data/src_unigram6000/spm_unigram_ko.txt, data/tgt_unigram6000/spm_unigram_en.txt

Place the assets according to the paths referenced in configs/echostream_config.yaml and configs/echostream_config.mini.yaml. If your paths differ, update the config files accordingly.

6) Configuration

- Inference config: configs/echostream_config.yaml
  - Points to global_cmvn, dictionaries, mHuBERT model, and (optional) vocoder paths
  - Sets streaming.ctc_threshold, and the server reads vocoder code_hop_size from the vocoder config.json
- Training (mini) config: configs/echostream_config.mini.yaml
  - Smaller Emformer/decoders
  - Emphasis on unit learning (multitask.unit_weight = 0.70)
  - data.units_root and data.load_tgt_units: true
  - streaming.ctc_threshold: 0.6
  - Uses train_sampled.units.tsv/dev_sampled.units.tsv/test_sampled.tsv by default

7) Quick health check

```bash
python -c "import torch, numpy; print('OK: torch', torch.__version__)"
```

8) Train (mini) - optional but recommended for quality

CPU-friendly run (DataLoader single worker to avoid shm issues):

```bash
source .venv/bin/activate
python scripts/train.py \
  --config configs/echostream_config.mini.yaml \
  --train-manifest /Users/you/EchoStream/data/train_sampled.units.tsv \
  --dev-manifest   /Users/you/EchoStream/data/dev_sampled.units.tsv \
  --save-dir checkpoints_mini \
  --num-workers 0
```

Notes:
- If you see torchcodec ImportError during data loading, the fallback in datasets/s2st_dataset.py uses librosa/soundfile automatically.
- If you still encounter shared memory errors on macOS, ensure num_workers=0 as above.

9) Start the server

```bash
source .venv/bin/activate
uvicorn server.fastapi_app:app --host 0.0.0.0 --port 8000
```

Endpoints:
- GET /health -> {"status":"ok"}
- GET /config -> current effective config
- Web UI -> http://127.0.0.1:8000/ (served from server/static/index.html)

10) Web UI (browser)

- Open http://127.0.0.1:8000/
- Click 녹음 시작 to stream microphone audio
- The server returns synthesized translated audio (raw PCM), played in the page

11) Python WebSocket client (optional)

```bash
source .venv/bin/activate
python server/client_ws.py --host 127.0.0.1 --port 8000
```

12) Streaming-quality features

Already implemented:
- MT incremental state: decoder state persists across chunks
- CTC gating & whole-word policy: use ST-CTC confidence to gate updates
- Duration hop sync: server reads code_hop_size from vocoder config.json and extracts only the new audio segment based on durations

13) Unit learning rationale

For natural prosody and low-latency stability, the model predicts target unit sequences (mHuBERT K-means) in addition to text. This reduces pop/mix artifacts and improves chunk boundary stability. Use the *.units.tsv manifests to enable unit supervision.

14) Troubleshooting

- “TorchCodec is required” during torchaudio.load:
  - Fixed by the built-in fallback: datasets/s2st_dataset.py tries torchaudio first, then soundfile/librosa.
- SHM/permissions on macOS:
  - Run training with --num-workers 0.
- Vocoder mismatch / pop-pop sounds:
  - Ensure vocoder config.json/g_*.pt match the unit settings (layer=11, km=1000) and that server uses duration prediction. Confirm code_hop_size alignment via the vocoder config.
- Unit loss stays 0:
  - Make sure you are using the *.units.tsv manifests and data.load_tgt_units: true.

15) Migrating to a new machine

1. Clone (shallow): 
   ```bash
   git clone --depth 1 --branch feature/streaming-mini-units https://github.com/Kyle-Riss/EchoStream.git
   cd EchoStream
   ```
2. Setup venv + requirements: 
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Copy assets to the paths referenced in configs (or update the configs):
   - pretrain_models/ (mHuBERT, vocoder)
   - data/ (tsv manifests, units/*.npy, gcmvn.npz, src/tgt dicts)
4. Optional: Mini training run to adapt
5. Start server and open the web UI

16) Optional: Generating units

If you need to (re-)generate units for your target set:
- Use your mHuBERT model and K-means to produce text unit files, then convert per-utterance to .npy (we include a conversion snippet in prior discussions; integrate as needed).
- Update the *.units.tsv manifests to point to the produced .npy files in data/units/.

17) Contributing

- Create a feature branch and open a PR (we used feature/streaming-mini-units as a template branch).
- Avoid committing data/units/checkpoints; prefer release artifacts or internal storage.

18) License

Please refer to the repository license. External models (mHuBERT, CodeHiFiGAN, etc.) follow their respective licenses.

# EchoStream 🎤→🗣️

**EchoStream: Efficient Memory-based Streaming Speech-to-Speech Translation**

EchoStream은 Emformer 기반의 고효율 실시간 음성-음성 번역 모델입니다. StreamSpeech 아키텍처를 기반으로 하되, Chunk-based Conformer 인코더를 **Emformer**로 교체하여 계산 효율성과 처리 속도를 크게 향상시켰습니다.

---

## ✨ 주요 특징

### 🚀 효율성 향상
- **Left Context Cache**: 이전 세그먼트의 Key/Value를 캐시하여 재사용
- **Augmented Memory Bank**: 장거리 의존성을 효율적으로 모델링
- **연산 복잡도**: O(T²) → O(1) (발화 길이와 무관하게 일정)

### ⚡ 성능 향상
- **속도**: 기존 대비 6-50배 빠름 (발화 길이에 따라)
- **메모리**: 25배 절약
- **지연 시간**: 일정한 낮은 지연 (발화 길이 무관)

### 🎯 실시간 번역
- **스트리밍 처리**: 청크 단위 실시간 번역
- **CT-Transformer 통합**: 구두점 기반 문장 경계 탐지 및 재조합
- **낮은 지연**: 10ms 수준의 인코더 지연

---

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────┐
│              EchoStream Architecture                 │
├─────────────────────────────────────────────────────┤
│ Speech Input → Emformer Encoder (16L)                │
│                    ↓                                 │
│        ┌───────────┴───────────┐                    │
│        ↓                       ↓                    │
│  ASR CTC Decoder      ST CTC Decoder                │
│        ↓                       ↓                    │
│  CT-Transformer    MT Decoder (4L)                  │
│        ↓                       ↓                    │
│  Sentence Boundary   T2U Encoder (0L)               │
│        ↓                       ↓                    │
│  Recomposition    Unit Decoder (6L)                 │
│        ↓                       ↓                    │
│    Output ←──── CodeHiFiGAN Vocoder                │
└─────────────────────────────────────────────────────┘
```

### 핵심 컴포넌트

1. **Emformer Encoder**: 효율적인 메모리 기반 스트리밍 인코더
2. **CTC Decoders**: ASR 및 ST (Speech-to-Text) 작업용
3. **MT Decoder**: 고품질 텍스트 번역
4. **Unit Decoder**: 텍스트를 음성 유닛으로 변환
5. **CodeHiFiGAN**: 유닛을 고품질 오디오로 합성
6. **CT-Transformer**: 실시간 구두점 예측 및 문장 경계 탐지

---

## 📊 성능 비교

| 메트릭 | StreamSpeech (Conformer) | EchoStream (Emformer) | 개선 |
|--------|-------------------------|----------------------|------|
| **인코더 지연** (10초 발화) | ~60ms | ~10ms | **6배** ⚡ |
| **메모리 사용량** | ~256MB | ~10MB | **25배** 💾 |
| **연산 복잡도** | O(T²) | O(1) | **일정** 🚀 |
| **처리 속도** | 발화 길이↑ → 느려짐 | 발화 길이 무관 | **일정** ✅ |

---

## 🚀 시작하기

### 설치

```bash
# 저장소 클론
git clone https://github.com/Kyle-Riss/Ko-Speech2Speech.git
cd StreamSpeech

# 의존성 설치
pip install -r requirements.txt
```

### 빠른 시작

```bash
# 추론 실행
python demo/infer.py \
    --model-path /path/to/model \
    --audio-path /path/to/audio.wav \
    --config configs/fr-en/config_unity.yaml
```

---

## 📚 문서

- [Emformer 통합 계획](EMFORMER_INTEGRATION_PLAN.md): Emformer 인코더 통합 상세 계획
- [CT-Transformer 통합](README_CT_TRANSFORMER_INTEGRATION.md): 구두점 예측 및 재조합 시스템
- [핵심 파일 가이드](CORE_FILES_FOR_REALTIME_TRANSLATION.md): 실시간 번역 관련 파일 정리

---

## 🔬 기반 연구

EchoStream은 다음 연구를 기반으로 합니다:

- **StreamSpeech**: [Streaming Speech-to-Speech Translation](https://arxiv.org/abs/2212.05758)
- **Emformer**: [Efficient Memory Transformer for Streaming ASR](https://arxiv.org/abs/2010.10759)
- **CT-Transformer**: [Controllable Time-Delay Transformer](https://ieeexplore.ieee.org/document/9054256)

---

## 📝 라이선스

본 프로젝트는 원본 StreamSpeech 및 Fairseq의 라이선스를 따릅니다.

---

## 🤝 기여

기여를 환영합니다! 이슈와 풀 리퀘스트를 통해 참여해 주세요.

---

## 📧 문의

- Repository: [https://github.com/Kyle-Riss/Ko-Speech2Speech](https://github.com/Kyle-Riss/Ko-Speech2Speech)

---

**EchoStream** - 빠르고 효율적인 실시간 음성-음성 번역 🌊
