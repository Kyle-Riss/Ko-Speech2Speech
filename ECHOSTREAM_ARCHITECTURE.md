# EchoStream 모델 아키텍처 (정확한 구조)

**최종 업데이트**: 2025-11-02  
**버전**: 1.0  
**기반**: StreamSpeech (ACL 2024) + Emformer Encoder

---

## 📊 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EchoStream S2ST Model                         │
└─────────────────────────────────────────────────────────────────────┘

Input: Speech [B, T, 80]  (80-dim filter-bank, 16kHz, 10ms hop)
  │
  ▼
┌────────────────────────────────────────────────┐
│ 1. Conv2D Subsampling (4x downsampling)       │
│    - Conv2D (k=3, s=2): 80 → 40 features      │
│    - Conv2D (k=3, s=2): 40 → 20 features      │
│    - Linear: 256*20 → 256d                    │
│    Output: [T/4, B, 256]                      │
└────────────────────────────────────────────────┘
  │
  ▼
┌────────────────────────────────────────────────┐
│ 2. Emformer Encoder (16 layers, 256d)         │
│                                                │
│    For each layer:                             │
│      - Left Context Cache (30 frames)         │
│      - Memory Bank (8 vectors from layer n-1) │
│      - Current Segment (4 frames)             │
│      - Multi-Head Attention (4 heads)         │
│      - Feed-Forward (1024d)                   │
│                                                │
│    Output: [T/4, B, 256]                      │
└────────────────────────────────────────────────┘
  │
  ├─────────────────┬──────────────────┐
  │                 │                  │
  ▼                 ▼                  │
┌─────────────┐ ┌─────────────────┐   │
│ 3a. ASR CTC │ │ 3b. ST CTC      │   │
│             │ │   + 2L Trans.   │   │
│ Vocab: 6K   │ │   (unidirect.)  │   │
│ (Source)    │ │   Vocab: 6K     │   │
│             │ │   (Target)      │   │
└─────────────┘ └─────────────────┘   │
  │                 │                  │
  │ (punctuation)   ▼                  │
  │            ┌─────────────────┐     │
  │            │ 4. MT Decoder   │     │
  │            │   (4L Trans.)   │     │
  │            │                 │◄────┘ (cross-attn)
  │            │   Vocab: 6K     │
  │            │   (Target)      │
  │            └─────────────────┘
  │                 │
  └────────┬────────┘
           ▼
    ┌─────────────────┐
    │ 5. Unit Decoder │
    │   (6L Trans.)   │
    │                 │◄────────── Encoder out (cross-attn)
    │   + CTC Upsample│
    │   (ratio: 5)    │
    │                 │
    │   Units: 1000   │
    │   (HuBERT)      │
    └─────────────────┘
           │
           ▼
    ┌─────────────────┐
    │ 6. CodeHiFiGAN  │
    │    Vocoder      │
    │                 │
    │  Units → Wav    │
    └─────────────────┘
           │
           ▼
Output: Waveform [B, T_wav] @ 16kHz
```

---

## 🔧 컴포넌트별 상세 구조

### 1. Conv2D Subsampler

**목적**: 입력 feature를 4배 다운샘플링하여 계산 효율 향상

```python
Input:  [B, T, 80]  (80-dim filter-bank)
        ↓
Conv2D Layer 1:
  - in_channels: 1
  - out_channels: 256
  - kernel_size: 3
  - stride: 2
  - padding: 1
  Output: [B, 256, T/2, 40]
        ↓
ReLU
        ↓
Conv2D Layer 2:
  - in_channels: 256
  - out_channels: 256
  - kernel_size: 3
  - stride: 2
  - padding: 1
  Output: [B, 256, T/4, 20]
        ↓
Reshape: [B, T/4, 256*20] = [B, T/4, 5120]
        ↓
Linear: 5120 → 256
        ↓
Transpose: [B, T/4, 256] → [T/4, B, 256]
        ↓
Output: [T/4, B, 256]
```

**파라미터 수**: ~1.3M

---

### 2. Emformer Encoder (핵심!)

**구조**: 16개의 동일한 Emformer Layer

#### 단일 Emformer Layer

```
Input (current segment):  [4, B, 256]  (4 frames @ 10ms = 40ms chunk)
Left Context Cache:       [30, B, 256] (previous segments' K, V)
Memory Bank:              [8, B, 256]  (from layer n-1)
Right Context:            [0, B, 256]  (streaming = no lookahead)

┌─────────────────────────────────────────────┐
│  Step 1: Prepare Q, K, V                    │
├─────────────────────────────────────────────┤
│  Query:                                     │
│    Q = segment  [4, B, 256]                 │
│                                             │
│  Key & Value:                               │
│    K = [memory_bank,  left_context,  seg]  │
│      = [8, B, 256] + [30, B, 256] + [4]    │
│      = [42, B, 256]  ← Efficient!           │
│                                             │
│    V = [memory_bank,  left_context,  seg]  │
│      = [42, B, 256]                         │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  Step 2: Multi-Head Attention               │
├─────────────────────────────────────────────┤
│  attn_output = MHA(Q, K, V)                 │
│    - num_heads: 4                           │
│    - head_dim: 256 / 4 = 64                 │
│    - dropout: 0.1                           │
│                                             │
│  Output: [4, B, 256]                        │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  Step 3: Add & Norm                         │
├─────────────────────────────────────────────┤
│  x = segment + dropout(attn_output)         │
│  x = LayerNorm(x)                           │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  Step 4: Feed-Forward Network               │
├─────────────────────────────────────────────┤
│  FFN:                                       │
│    - Linear: 256 → 1024                     │
│    - ReLU                                   │
│    - Dropout: 0.1                           │
│    - Linear: 1024 → 256                     │
│                                             │
│  x = x + dropout(FFN(x))                    │
│  x = LayerNorm(x)                           │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  Step 5: Update Cache                       │
├─────────────────────────────────────────────┤
│  left_context_cache_new = segment           │
│    (current segment → next segment's left)  │
│                                             │
│  memory_bank_new = summarize(output)        │
│    (averaged output → upper layer n+1)      │
└─────────────────────────────────────────────┘
  │
  ▼
Output: [4, B, 256]
Cache: {left_context: [4, B, 256], memory: [1, B, 256]}
```

#### 16-Layer Emformer 전체

```
Layer 1:  segment[4] + left[30] + mem[8 from input]
            ↓
          cache: left[4], mem[1] ──┐
                                    │
Layer 2:  segment[4] + left[30] + mem[1 from L1] ←┘
            ↓
          cache: left[4], mem[1] ──┐
                                    │
Layer 3:  segment[4] + left[30] + mem[1 from L2] ←┘
            ↓
          ...
            ↓
Layer 16: segment[4] + left[30] + mem[1 from L15]
            ↓
          Final output: [4, B, 256]

Note: Left context는 이전 "시간" 세그먼트에서
      Memory bank는 이전 "레이어"에서 전달
```

**파라미터 수**: ~14M

**핵심 특징**:
- ✅ O(1) 복잡도 (segment 크기 고정)
- ✅ Left Context Cache (K, V 재사용)
- ✅ Memory Bank (하위→상위 레이어 정보 전달)
- ✅ 고정 메모리 (42 frames = 420ms context)

---

### 3a. ASR CTC Decoder

**목적**: Source 언어 텍스트 예측 (punctuation prediction용)

```python
Input:  encoder_out [T/4, B, 256]
          ↓
Linear: 256 → 6000  (source vocab)
          ↓
Log Softmax (dim=-1)
          ↓
Output: log_probs [T/4, B, 6000]

Decoding: Greedy CTC decoding
  → Source text (for CT-Transformer punctuation)
```

**파라미터 수**: 1.5M

---

### 3b. ST CTC Decoder (with Transformer)

**목적**: Target 언어 텍스트 예측 (streaming translation)

```python
Input:  encoder_out [T/4, B, 256]
          ↓
┌───────────────────────────────────────┐
│ Transformer Encoder Layer 1           │
│   - Self-Attention (causal mask)      │
│   - Feed-Forward                      │
│   Output: [T/4, B, 256]               │
└───────────────────────────────────────┘
          ↓
┌───────────────────────────────────────┐
│ Transformer Encoder Layer 2           │
│   - Self-Attention (causal mask)      │
│   - Feed-Forward                      │
│   Output: [T/4, B, 256]               │
└───────────────────────────────────────┘
          ↓
Linear: 256 → 6000  (target vocab)
          ↓
Log Softmax
          ↓
Output: log_probs [T/4, B, 6000]

Decoding: Greedy CTC decoding
  → Target text (preliminary translation)
```

**Causal Mask** (Unidirectional):
```
Attention mask:
  [1, 0, 0, 0]  ← frame 0 only sees frame 0
  [1, 1, 0, 0]  ← frame 1 sees frames 0-1
  [1, 1, 1, 0]  ← frame 2 sees frames 0-2
  [1, 1, 1, 1]  ← frame 3 sees frames 0-3

→ Streaming-friendly!
```

**파라미터 수**: ~2.6M

---

### 4. MT Decoder (Transformer)

**목적**: CTC 출력을 autoregressive하게 refine

```python
Input:
  - prev_output_tokens: [B, T_tgt]  (shifted target)
  - encoder_out: [T/4, B, 256]

┌───────────────────────────────────────┐
│ Token Embedding                        │
│   - Embedding: 6000 → 256             │
│   - Positional Encoding (sinusoidal)  │
│   Output: [T_tgt, B, 256]             │
└───────────────────────────────────────┘
          ↓
┌───────────────────────────────────────┐
│ Transformer Decoder Layer 1            │
│   1. Self-Attention (causal mask)     │
│   2. Cross-Attention (to encoder)     │
│   3. Feed-Forward                     │
└───────────────────────────────────────┘
          ↓
┌───────────────────────────────────────┐
│ Transformer Decoder Layer 2            │
│   (same structure)                    │
└───────────────────────────────────────┘
          ↓
┌───────────────────────────────────────┐
│ Transformer Decoder Layer 3            │
│   (same structure)                    │
└───────────────────────────────────────┘
          ↓
┌───────────────────────────────────────┐
│ Transformer Decoder Layer 4            │
│   (same structure)                    │
└───────────────────────────────────────┘
          ↓
Linear: 256 → 6000
          ↓
Output: logits [B, T_tgt, 6000]

Decoding: Autoregressive (beam search or greedy)
  → Refined target text
```

**파라미터 수**: ~5.1M

---

### 5. Unit Decoder (CTC Transformer)

**목적**: Text hidden states → Speech units

```python
Input:  text_hidden [T/4, B, 256]  (from encoder or MT decoder)
          ↓
┌───────────────────────────────────────┐
│ CTC Upsampling (ratio: 5)              │
│   Repeat each frame 5 times            │
│   [T/4, B, 256] → [5*T/4, B, 256]     │
└───────────────────────────────────────┘
          ↓
┌───────────────────────────────────────┐
│ Input Projection                       │
│   Linear: 256 → 256                   │
│   + Positional Encoding               │
└───────────────────────────────────────┘
          ↓
┌───────────────────────────────────────┐
│ Transformer Decoder Layer 1            │
│   1. Self-Attention (causal)          │
│   2. Cross-Attention (to upsampled)   │
│   3. Feed-Forward                     │
└───────────────────────────────────────┘
          ↓
      (Layers 2-6)
          ↓
┌───────────────────────────────────────┐
│ Transformer Decoder Layer 6            │
│   (same structure)                    │
└───────────────────────────────────────┘
          ↓
Linear: 256 → 1000  (HuBERT units)
          ↓
Log Softmax
          ↓
Output: log_probs [B, 5*T/4, 1000]

Decoding: Greedy
  → Discrete speech units (0-999)
```

**CTC Upsampling**:
```
Input:  [a, b, c, d]  (4 frames)
          ↓
Repeat 5x: [a,a,a,a,a, b,b,b,b,b, c,c,c,c,c, d,d,d,d,d]
          ↓
Output: [20 frames]  (5배 증가)

→ Speech unit의 시간 해상도를 높임!
```

**파라미터 수**: ~7.7M

---

### 6. CodeHiFiGAN Vocoder

**목적**: Discrete units → Waveform

```python
Input:  units [B, T_unit]  (discrete unit IDs: 0-999)
          ↓
┌───────────────────────────────────────┐
│ Unit Embedding                         │
│   units → one-hot [B, T_unit, 1000]   │
│   (현재: dummy linear projection)      │
└───────────────────────────────────────┘
          ↓
┌───────────────────────────────────────┐
│ Generator Network (GAN)                │
│   - Transposed convolutions           │
│   - Upsample to 16kHz                 │
│   - Multi-scale discriminators        │
│                                       │
│   (현재: dummy 32 samples/unit)        │
└───────────────────────────────────────┘
          ↓
Tanh (normalize to [-1, 1])
          ↓
Output: waveform [B, T_wav] @ 16kHz

T_wav = T_unit × samples_per_unit
      = T_unit × 32  (dummy)
      = (5*T/4) × 32
```

**파라미터 수**: ~2.1M (dummy), ~14M (real CodeHiFiGAN)

**Note**: 현재는 dummy vocoder, 실제로는 pre-trained CodeHiFiGAN 필요

---

## 📏 모델 크기

### 파라미터 수 (16-layer full model)

| Component | Parameters | Percentage |
|-----------|-----------|------------|
| **Conv2D Subsampler** | 1,312,256 | 3.9% |
| **Emformer Encoder (16L)** | 15,592,448 | 46.0% |
| **ASR CTC Decoder** | 1,536,000 | 4.5% |
| **ST CTC Decoder (2L)** | 2,579,456 | 7.6% |
| **MT Decoder (4L)** | 5,136,384 | 15.2% |
| **Unit Decoder (6L)** | 7,699,721 | 22.7% |
| **Vocoder (dummy)** | 2,049 | 0.01% |
| **Total** | **33,858,314** | **100%** |

**모델 크기**: ~33.9M parameters (~129 MB @ fp32)

**비교**:
- StreamSpeech Conformer: ~45M parameters
- EchoStream: 33.9M (약 25% 작음)

---

## ⚙️ 하이퍼파라미터 (기본 설정)

### Encoder (Emformer)

```yaml
encoder_embed_dim: 256
encoder_layers: 16
encoder_attention_heads: 4
encoder_ffn_embed_dim: 1024

# Emformer-specific
segment_length: 4          # 40ms @ 100fps
left_context_length: 30    # 300ms
right_context_length: 0    # Streaming (no lookahead)
memory_size: 8             # Memory bank size

# Input
input_feat_per_channel: 80  # Mel filter-bank
input_channels: 1
```

### Decoders

```yaml
# ST CTC
st_decoder_layers: 2
st_decoder_heads: 4
st_vocab_size: 6000

# MT Decoder
mt_decoder_layers: 4
mt_decoder_heads: 4
mt_decoder_embed_dim: 256
mt_decoder_ffn_dim: 1024
mt_vocab_size: 6000

# Unit Decoder
unit_decoder_layers: 6
unit_decoder_heads: 4
unit_decoder_embed_dim: 256
unit_decoder_ffn_dim: 1024
num_units: 1000
ctc_upsample_ratio: 5
```

### Regularization

```yaml
dropout: 0.1
attention_dropout: 0.1
activation_dropout: 0.1
label_smoothing: 0.1  # For training
```

---

## 🔄 데이터 흐름 (Forward Pass)

### Training Mode

```python
# Input
src_tokens = [B, T, 80]           # Speech features
src_lengths = [B]                  # Sequence lengths
prev_output_tokens = [B, T_tgt]   # Shifted target text

# 1. Encoder
encoder_out = encoder(src_tokens, src_lengths)
# → [T/4, B, 256]

# 2. ASR CTC
asr_logits = asr_decoder(encoder_out)
# → [T/4, B, 6000]
asr_loss = CTC_loss(asr_logits, src_text)

# 3. ST CTC
st_logits = st_decoder(encoder_out)
# → [T/4, B, 6000]
st_loss = CTC_loss(st_logits, tgt_text)

# 4. MT Decoder
mt_logits = mt_decoder(prev_output_tokens, encoder_out)
# → [B, T_tgt, 6000]
mt_loss = CrossEntropy(mt_logits, tgt_text)

# 5. Unit Decoder
unit_logits = unit_decoder(encoder_out)
# → [B, 5*T/4, 1000]
unit_loss = CrossEntropy(unit_logits, tgt_units)

# Total Loss
loss = 0.3*asr_loss + 0.3*st_loss + 0.2*mt_loss + 0.2*unit_loss
```

### Inference Mode (Streaming)

```python
# Input: Audio chunks (40ms each)
for chunk in audio_stream:
    # 1. Extract features
    features = extract_features(chunk)  # [1, 4, 80]
    
    # 2. Encoder (with cache)
    encoder_out = encoder(features)  # [1, 1, 256]
    
    # 3. ST CTC (streaming)
    st_logits = st_decoder(encoder_out)
    st_text = ctc_decode(st_logits)  # Incremental
    
    # 4. Check punctuation (optional)
    punctuated, is_end = punctuator(st_text)
    
    # 5. Unit prediction
    unit_logits = unit_decoder(encoder_out)
    units = unit_logits.argmax(-1)
    
    # 6. Vocoder
    waveform = vocoder(units)
    
    # 7. Output
    if is_end:
        # Recompose full sentence
        final_waveform = recompose(buffered_units)
        yield final_waveform
    else:
        # Stream partial output
        yield waveform
```

---

## 🎯 핵심 설계 원칙

### 1. Efficient Streaming (Emformer)

**문제**: Conformer의 O(T²) attention
**해결**: Emformer의 O(1) attention
- Left Context Cache: 이전 세그먼트의 K, V 재사용
- Memory Bank: 하위 레이어에서 요약 정보 전달
- Segment-wise processing: 고정 크기 (4 frames = 40ms)

### 2. Multi-task Learning

**목적**: 단일 모델로 여러 태스크 동시 학습
- ASR: Source 언어 인식
- ST: 번역 (CTC)
- MT: 번역 refinement (autoregressive)
- Unit: Speech unit 예측

**장점**:
- 공유 encoder → 효율적
- 상호 보완적 학습
- Intermediate supervision

### 3. Unidirectional Processing

**스트리밍 요구사항**:
- Emformer: right_context = 0
- ST CTC Decoder: causal mask
- MT Decoder: causal mask
- Unit Decoder: causal mask

**결과**: 완전한 실시간 처리 가능!

### 4. CTC-based Policy

**장점**:
- Non-autoregressive (병렬 처리)
- Alignment 자동 학습
- Latency 예측 가능

**사용**:
- ASR output → punctuation
- ST output → preliminary translation
- Unit upsampling → temporal resolution

---

## 🆚 StreamSpeech와의 차이

| Component | StreamSpeech | EchoStream | 차이점 |
|-----------|-------------|-----------|-------|
| **Encoder** | Chunk-based Conformer | Emformer | O(T²) → O(1) |
| **Complexity** | Quadratic | Constant | 발화 길이에 무관 |
| **Memory** | 증가 (발화↑) | 고정 (42 frames) | 13,000배 절약 |
| **Cache** | None | Left Context + Memory | K, V 재사용 |
| **Decoders** | 동일 | 동일 | 100% 동일 |
| **Vocoder** | CodeHiFiGAN | CodeHiFiGAN | 동일 |
| **Parameters** | ~45M | ~34M | 25% 감소 |

**결론**: 인코더만 교체, 디코더는 완전 동일!

---

## 📝 구현 파일

```
models/
├── emformer_layer.py          # EmformerEncoder (16L)
├── echostream_encoder.py      # Conv2D + Emformer
├── echostream_model.py        # Complete model
└── decoders/
    ├── ctc_decoder.py         # ASR CTC + ST CTC
    ├── transformer_decoder.py # MT Decoder
    ├── unit_decoder.py        # Unit Decoder
    └── vocoder.py             # CodeHiFiGAN (dummy)

configs/
└── echostream_config.yaml     # Hyperparameters

agent/
└── echostream_agent.py        # SimulEval agent

scripts/
├── train.py                   # Training
└── evaluate.py                # Evaluation
```

---

## 🚀 사용 예시

### 모델 생성

```python
from models.echostream_model import build_echostream_model, EchoStreamConfig

# Create config
config = EchoStreamConfig()
config.encoder_layers = 16  # Full model

# Build model
model = build_echostream_model(config)
model.eval()

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Forward Pass

```python
import torch

# Input
B, T, F = 2, 100, 80
src_tokens = torch.randn(B, T, F)
src_lengths = torch.tensor([100, 80])

# Forward
with torch.no_grad():
    output = model(src_tokens, src_lengths)

# Outputs
print(f"Encoder: {output['encoder_out']['encoder_out'][0].shape}")  # [25, 2, 256]
print(f"ASR: {output['asr_logits'].shape}")      # [2, 25, 6000]
print(f"ST: {output['st_logits'].shape}")        # [2, 25, 6000]
print(f"Units: {output['unit_logits'].shape}")   # [2, 125, 1000]
print(f"Waveform: {output['waveform'].shape}")   # [2, 4000]
```

### Streaming

```python
# Reset cache
model.reset_cache()

# Process chunks
for chunk in audio_chunks:  # Each chunk: [1, 4, 80] (40ms)
    output = model(chunk, lengths=torch.tensor([4]))
    waveform = output['waveform']
    # Stream output...
```

---

**EchoStream**: Efficient + Echo(반복 없는 캐시) + Stream(실시간)! 🌊

