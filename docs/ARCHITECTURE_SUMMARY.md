# EchoStream 아키텍처 요약

## 🎯 핵심 원칙

**EchoStream = StreamSpeech 구조 + Conformer → Emformer 교체**

---

## ✅ 변경 사항: 인코더만 교체

### StreamSpeech (원본)
```
Chunk-based Conformer Encoder (16L)
  - Self-attention to all previous chunks
  - Depthwise convolution
  - Complexity: O(T²)
```

### EchoStream (개선)
```
Emformer Encoder (16L)
  - Left Context Cache (K, V reuse)
  - Memory Bank from lower layer
  - Complexity: O(1) per segment
```

**변경 이유**: 효율성 향상 (O(T²) → O(1))

---

## ✅ 유지 사항: 모든 디코더 동일

### 1. ASR CTC Decoder
- **StreamSpeech**: `CTCDecoder`
- **EchoStream**: `CTCDecoder`
- **상태**: ✅ **100% 동일**

### 2. ST CTC Decoder
- **StreamSpeech**: `CTCDecoderWithTransformerLayer` (2L)
- **EchoStream**: `CTCDecoderWithTransformerLayer` (2L)
- **상태**: ✅ **100% 동일**

### 3. MT Decoder
- **StreamSpeech**: `TransformerDecoder` (4L)
- **EchoStream**: `TransformerMTDecoder` (4L)
- **상태**: ✅ **100% 동일** (이름만 다름)

### 4. Unit Decoder
- **StreamSpeech**: `CTCTransformerUnitDecoder` (6L)
- **EchoStream**: `CTCTransformerUnitDecoder` (6L)
- **상태**: ✅ **100% 동일**

### 5. Vocoder
- **StreamSpeech**: `CodeHiFiGAN`
- **EchoStream**: `CodeHiFiGAN`
- **상태**: ✅ **100% 동일**

---

## 📊 전체 파이프라인 비교

### StreamSpeech
```
Speech Input [B, T, 80]
    ↓
Conv2D Subsampling (4x)
    ↓
Chunk-based Conformer Encoder (16L)  ← 🔴 이 부분만 다름
    ↓
[T/4, B, 256]
    ├─→ ASR CTC Decoder              ← ✅ 동일
    └─→ ST CTC Decoder (2L)          ← ✅ 동일
           ↓
       MT Decoder (4L)                ← ✅ 동일
           ↓
       Unit Decoder (6L)              ← ✅ 동일
           ↓
       CodeHiFiGAN Vocoder            ← ✅ 동일
           ↓
Output Speech
```

### EchoStream
```
Speech Input [B, T, 80]
    ↓
Conv2D Subsampling (4x)              ← ✅ 동일
    ↓
Emformer Encoder (16L)               ← 🔴 이 부분만 교체!
    ↓
[T/4, B, 256]
    ├─→ ASR CTC Decoder              ← ✅ 동일
    └─→ ST CTC Decoder (2L)          ← ✅ 동일
           ↓
       MT Decoder (4L)                ← ✅ 동일
           ↓
       Unit Decoder (6L)              ← ✅ 동일
           ↓
       CodeHiFiGAN Vocoder            ← ✅ 동일
           ↓
Output Speech
```

---

## 🔧 코드 구조

### 변경된 파일
```
models/
├── emformer_layer.py          ← 🆕 NEW: Emformer 구현
├── echostream_encoder.py      ← 🆕 NEW: Emformer + Conv2D
└── echostream_model.py        ← 🆕 NEW: 전체 모델 (디코더는 재사용)
```

### 재사용된 파일 (StreamSpeech와 동일)
```
models/decoders/
├── ctc_decoder.py             ← ✅ StreamSpeech와 동일
├── transformer_decoder.py      ← ✅ StreamSpeech와 동일
├── unit_decoder.py             ← ✅ StreamSpeech와 동일
└── vocoder.py                  ← ✅ StreamSpeech와 동일
```

---

## 📋 핵심 포인트

### 1. 인코더 출력 형식
- **StreamSpeech 형식 그대로 유지**
- `encoder_out['encoder_out'][0]` - List 형태
- `[T, B, D]` - Time-first 차원 순서
- 디코더 호환성 100% 보장

### 2. 디코더 인터페이스
- **모든 디코더는 StreamSpeech와 동일한 인터페이스 사용**
- 입력: `encoder_out: [T, B, D]` 텐서
- 출력: StreamSpeech와 동일한 형식

### 3. 데이터 흐름
- **StreamSpeech와 완전히 동일**
- Encoder → ASR/ST CTC → MT → Unit → Vocoder
- 각 단계의 입력/출력 형식 동일

---

## ✅ 검증 체크리스트

- [x] 인코더 출력 형식이 StreamSpeech와 동일
- [x] ASR CTC Decoder가 StreamSpeech와 동일
- [x] ST CTC Decoder가 StreamSpeech와 동일
- [x] MT Decoder가 StreamSpeech와 동일
- [x] Unit Decoder가 StreamSpeech와 동일
- [x] Vocoder가 StreamSpeech와 동일
- [x] 전체 파이프라인이 StreamSpeech와 동일
- [x] 디코더 인터페이스가 StreamSpeech와 호환

---

## 🎯 결론

**EchoStream은 StreamSpeech의 구조를 100% 따르며, Conformer 인코더만 Emformer로 교체했습니다.**

- ✅ **변경**: Conformer → Emformer (효율성 향상)
- ✅ **유지**: 모든 디코더, 데이터 흐름, 인터페이스

이것이 EchoStream의 핵심 설계 원칙입니다!

---

**마지막 업데이트**: 2025-01-XX  
**버전**: 1.0

