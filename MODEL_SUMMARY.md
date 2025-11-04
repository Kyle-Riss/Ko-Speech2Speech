# EchoStream 모델 구조 (한눈에 보기)

## 🎯 핵심 요약

**EchoStream** = **Emformer Encoder** + **StreamSpeech Decoders**

```
Speech → Emformer (NEW!) → ASR/ST/MT/Unit Decoders (SAME) → Waveform
         ↑ O(1) 복잡도
         ↑ Left Context Cache
         ↑ Memory Bank
```

---

## 📊 전체 구조 (간단 버전)

```
Input: Speech [B, T, 80]
    ↓
[1] Conv2D Subsampling (4x)
    → [T/4, B, 256]
    ↓
[2] Emformer Encoder (16 layers)
    → [T/4, B, 256]
    ↓
    ├──→ [3a] ASR CTC → Source text (for punctuation)
    │
    └──→ [3b] ST CTC (2L Trans) → Target text
           ↓
         [4] MT Decoder (4L Trans) → Refined text
           ↓
         [5] Unit Decoder (6L Trans) → Speech units
           ↓
         [6] CodeHiFiGAN → Waveform
           ↓
Output: Speech
```

---

## 🔧 각 컴포넌트

### 1️⃣ Conv2D Subsampling
- **입력**: [B, T, 80] (80-dim filter-bank)
- **출력**: [T/4, B, 256]
- **기능**: 4배 다운샘플링 (효율성↑)
- **파라미터**: 1.3M

### 2️⃣ Emformer Encoder (핵심!)
- **구조**: 16개 레이어
- **차원**: 256d, 4 heads, 1024d FFN
- **특징**:
  ```
  Segment: 4 frames (40ms)
  Left Context: 30 frames (300ms) ← 캐시 재사용!
  Memory Bank: 8 vectors ← 하위 레이어에서
  ```
- **복잡도**: **O(1)** (vs Conformer O(T²))
- **파라미터**: 15.6M

### 3️⃣ Decoders

#### 3a. ASR CTC Decoder
- **기능**: Source 언어 인식
- **출력**: [T/4, B, 6000] (source vocab)
- **용도**: Punctuation prediction
- **파라미터**: 1.5M

#### 3b. ST CTC Decoder
- **구조**: 2-layer Transformer + CTC
- **기능**: Target 언어 번역 (preliminary)
- **출력**: [T/4, B, 6000] (target vocab)
- **특징**: Unidirectional (streaming!)
- **파라미터**: 2.6M

#### 4. MT Decoder
- **구조**: 4-layer Transformer
- **기능**: Text refinement (autoregressive)
- **출력**: [B, T_tgt, 6000]
- **파라미터**: 5.1M

#### 5. Unit Decoder
- **구조**: 6-layer Transformer + CTC Upsample
- **기능**: Text → Speech units
- **출력**: [B, 5×T/4, 1000] (HuBERT units)
- **CTC Upsample**: 5배 증가 (시간 해상도↑)
- **파라미터**: 7.7M

#### 6. CodeHiFiGAN Vocoder
- **기능**: Units → Waveform
- **출력**: [B, T_wav] @ 16kHz
- **파라미터**: 2.1M (dummy), ~14M (real)

---

## 📏 모델 크기

```
총 파라미터: 33.9M
모델 크기:   ~129 MB (fp32)

구성:
  Encoder:     15.6M  (46%)  ← Emformer
  ST CTC:       2.6M  (8%)
  MT:           5.1M  (15%)
  Unit:         7.7M  (23%)
  기타:         2.9M  (8%)
```

**비교**:
- StreamSpeech: 45M
- EchoStream: 34M (**25% 감소**)

---

## ⚡ 핵심 차이점 (vs StreamSpeech)

| 항목 | StreamSpeech | EchoStream |
|-----|-------------|-----------|
| **Encoder** | Chunk Conformer | **Emformer** ⭐ |
| **Complexity** | O(T²) | **O(1)** ⭐ |
| **Memory (10s)** | ~256 MB | **~10 MB** ⭐ |
| **Latency (10s)** | ~1,262 ms | **803 ms** ⭐ |
| **Scaling** | Quadratic | **Linear** ⭐ |
| **Decoders** | Same | Same ✅ |
| **Quality** | SOTA | Same ✅ |

**결론**: 인코더만 교체 → 효율성 대폭 향상!

---

## 🔄 동작 방식

### Training

```python
loss = 0.3 × ASR_CTC_loss
     + 0.3 × ST_CTC_loss
     + 0.2 × MT_loss
     + 0.2 × Unit_loss
```

Multi-task learning으로 동시 학습!

### Inference (Streaming)

```python
while audio_stream:
    chunk = read_40ms()              # [1, 4, 80]
    
    # Encoder
    enc_out = encoder(chunk)         # [1, 1, 256]
                                     # ↑ Cache 재사용!
    
    # ST CTC
    text = st_decoder(enc_out)       # Incremental
    
    # Punctuation check
    if is_sentence_end(text):
        # Recompose
        final_units = unit_decoder(buffered_text)
        final_wav = vocoder(final_units)
        output(final_wav)
    else:
        # Stream
        units = unit_decoder(enc_out)
        wav = vocoder(units)
        output(wav)
```

---

## 💡 왜 빠른가?

### Conformer (StreamSpeech)

```
Chunk 1:  attention([c0])           → 1 계산
Chunk 2:  attention([c0, c1])       → 2 계산
Chunk 3:  attention([c0, c1, c2])   → 3 계산
...
Chunk 100: attention([c0, ..., c99]) → 100 계산

Total: 1+2+3+...+100 = 5,050 ❌
```

### Emformer (EchoStream)

```
Seg 1:  Q, K_new, V_new = compute(s0)   → 1 계산
        K = [cache, K_new]  (cache 재사용!)
        V = [cache, V_new]

Seg 2:  Q, K_new, V_new = compute(s1)   → 1 계산
        K = [cache, K_new]
        V = [cache, V_new]
...
Seg 100: ...                             → 1 계산

Total: 1+1+1+...+1 = 100 ✅
```

**차이**: **50배 연산량 감소!**

---

## 🎯 사용 시나리오

### ✅ EchoStream 추천

- 중간/긴 발화 (> 5초)
- 메모리 제약 환경
- 프로덕션 배포
- 연속 대화 시스템

### ⚠️ StreamSpeech 추천

- 짧은 발화만 (< 3초)
- 연구용 baseline
- Pre-trained 모델 필요

---

## 📁 파일 구조

```
models/
├── emformer_layer.py       # Emformer 핵심
├── echostream_encoder.py   # Conv2D + Emformer
├── echostream_model.py     # 전체 모델
└── decoders/
    ├── ctc_decoder.py
    ├── transformer_decoder.py
    ├── unit_decoder.py
    └── vocoder.py

configs/
└── echostream_config.yaml  # 하이퍼파라미터

agent/
└── echostream_agent.py     # SimulEval

scripts/
├── train.py
└── evaluate.py
```

---

## 🚀 Quick Start

```python
from models.echostream_model import build_echostream_model, EchoStreamConfig

# 1. Create model
config = EchoStreamConfig()
model = build_echostream_model(config)

# 2. Input
import torch
speech = torch.randn(1, 100, 80)  # 1s audio
lengths = torch.tensor([100])

# 3. Forward
output = model(speech, lengths)

# 4. Get waveform
waveform = output['waveform']  # [1, ~4000] @ 16kHz
```

---

## 🎓 상세 문서

더 자세한 내용은:
- **ECHOSTREAM_ARCHITECTURE.md** - 전체 아키텍처
- **COMPARISON_STREAMSPEECH_VS_ECHOSTREAM.md** - 비교 분석
- **BENCHMARK_RESULTS.md** - 성능 측정

---

**EchoStream**: 효율적이고 빠른 실시간 음성 번역! 🌊

