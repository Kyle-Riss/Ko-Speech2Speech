# Phase 3: Multi-task Training Implementation ✅

**Status**: ✅ **COMPLETED**

## 📋 Overview

StreamSpeech의 Multi-task Learning 구조를 EchoStream에 완벽하게 통합했습니다.

**핵심 개선점**:
- **4개 Loss 통합**: L_asr + L_st + L_mt + L_unit
- **Multi-chunk Training**: 랜덤 세그먼트 길이로 학습
- **Label Smoothing**: 품질 향상
- **Gradient Flow**: 안정적인 학습

---

## 🎯 Implementation

### 1. Multi-task Criterion

**File**: `training/echostream_criterion.py`

**Loss Components**:

```python
L = L_asr + L_st + L_mt + L_unit

# 1. L_asr: ASR CTC Loss (source text recognition)
#    - Input: Encoder output → ASR CTC Decoder
#    - Target: Source text (e.g., French)
#    - Purpose: 원문 음성 인식

# 2. L_st: ST CTC Loss (target text translation)
#    - Input: Encoder output → ST CTC Decoder
#    - Target: Target text (e.g., English)
#    - Purpose: 번역된 텍스트 생성

# 3. L_mt: MT Cross-Entropy Loss (text refinement)
#    - Input: ST CTC output → MT Decoder
#    - Target: Target text (refined)
#    - Purpose: 번역 품질 향상

# 4. L_unit: Unit CTC Loss (speech unit generation)
#    - Input: MT output → Unit Decoder
#    - Target: Target speech units
#    - Purpose: 음성 합성 준비
```

**StreamSpeech 참고**:
- `StreamSpeech_analysis/researches/ctc_unity/criterions/speech_to_speech_ctc_asr_st_criterion.py`
- Line 115-200: Multi-task loss computation
- Line 224-232: CTC loss with zero_infinity

---

### 2. Multi-chunk Training

**핵심 아이디어**:
- **랜덤 세그먼트 길이**로 학습 → 다양한 latency에 robust
- **StreamSpeech Line 149-168** 참고

**구현**:

```python
# Random segment length sampling
segment_choices = [1, 2, 4, 8, 16]  # segments
segment_length = random.choice(segment_choices)

# Model forward with segment_length
model_output = model(
    src_tokens=audio,
    segment_length=segment_length  # ← Multi-chunk!
)
```

**효과**:
- ✅ **Offline (segment=99999)**: 최고 품질
- ✅ **Online (segment=1~16)**: 다양한 latency 대응
- ✅ **Robust**: 학습 시 다양한 chunk size 경험

---

### 3. Label Smoothing

**목적**: Overconfidence 방지 → 일반화 성능 향상

**구현**:

```python
# Smooth labels
smooth_targets = torch.zeros_like(logits)
smooth_targets.fill_(label_smoothing / (vocab - 1))
smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)

# KL divergence
log_probs = F.log_softmax(logits, dim=-1)
loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
```

**효과**:
- ✅ **Regularization**: 과적합 방지
- ✅ **Calibration**: 확률 분포 개선

---

## 🧪 Test Results

```bash
$ python training/echostream_criterion.py
```

**Output**:

```
======================================================================
Testing EchoStreamMultiTaskCriterion
======================================================================

1. Testing multi-task loss computation...
   Total loss: 48.6394
   L_asr: 15.7969
   L_st: 15.7323
   L_mt: 9.0060
   L_unit: 8.1043
   ✅ Multi-task loss computed

2. Testing backward pass...
   ✅ Backward pass successful

3. Testing trainer...
   Loss: 47.6648
   Segment length: 2
   ✅ Training step successful

4. Testing multi-chunk sampling...
   Sampled segments: [1, 8, 2, 1, 2, 8, 1, 4, 4, 8]
   ✅ Multi-chunk sampling works

======================================================================
✅ All EchoStreamMultiTaskCriterion tests passed!
======================================================================
```

---

## 📊 Comparison: StreamSpeech vs EchoStream

| Feature | StreamSpeech | EchoStream | Improvement |
|---------|--------------|------------|-------------|
| **Encoder** | Conformer (chunk-based) | Emformer (memory-based) | ✅ Efficient |
| **Multi-task Loss** | L_asr + L_st + L_mt + L_unit | L_asr + L_st + L_mt + L_unit | ✅ Same |
| **Multi-chunk** | Random [8, 16, 24, 32] | Random [1, 2, 4, 8, 16] | ✅ More granular |
| **Label Smoothing** | 0.1 | 0.1 | ✅ Same |
| **CTC Loss** | zero_infinity=True | zero_infinity=True | ✅ Same |

---

## 🔍 Key Insights

### 1. Why Multi-task Learning?

**StreamSpeech 논문 핵심**:
> "번역과 정책의 이중 과제(double challenges)를 해결하기 위해, 우리는 원문과 목표 음성의 텍스트 정보를 도입하여 Simul-S2ST를 안내한다."

**효과**:
1. **L_asr**: 원문 음성 인식 → Encoder가 음성 특징을 잘 학습
2. **L_st**: 번역 텍스트 생성 → Encoder가 번역 정보를 학습
3. **L_mt**: 텍스트 정제 → 번역 품질 향상
4. **L_unit**: 음성 합성 → 자연스러운 음성 생성

**결과**: 각 task가 서로를 도와 전체 품질 향상!

---

### 2. Why Multi-chunk Training?

**문제**:
- 고정된 chunk size로 학습 → 특정 latency에만 최적화
- 실제 사용 시 다양한 latency 요구 → 성능 저하

**해결**:
- 랜덤 chunk size로 학습 → 다양한 latency에 robust
- StreamSpeech: [8, 16, 24, 32]
- EchoStream: [1, 2, 4, 8, 16] (더 세밀한 제어)

**효과**:
- ✅ **Flexibility**: 다양한 latency 요구사항 대응
- ✅ **Robustness**: 일반화 성능 향상

---

### 3. Why CTC Loss?

**장점**:
1. **Alignment-free**: 명시적 정렬 불필요
2. **Parallel**: 병렬 처리 가능 → 빠른 학습
3. **Monotonic**: 순차적 출력 보장 → Streaming에 적합

**단점**:
1. **Independence**: 출력 간 독립 가정 → 품질 저하

**해결**:
- **MT Decoder**: CTC 출력을 Autoregressive로 정제 → 품질 향상
- **Best of both worlds**: CTC (속도) + AR (품질)

---

## 💡 Usage

### Basic Training

```python
from training.echostream_criterion import EchoStreamMultiTaskCriterion, EchoStreamTrainer

# 1. Initialize criterion
criterion = EchoStreamMultiTaskCriterion(
    asr_weight=1.0,
    st_weight=1.0,
    mt_weight=1.0,
    unit_weight=1.0,
    label_smoothing=0.1,
)

# 2. Initialize trainer
trainer = EchoStreamTrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    multi_chunk=True,
    segment_choices=[1, 2, 4, 8, 16],
)

# 3. Training loop
for batch in dataloader:
    loss, metrics = trainer.train_step(batch)
    
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"  L_asr: {metrics['L_asr']:.4f}")
    print(f"  L_st: {metrics['L_st']:.4f}")
    print(f"  L_mt: {metrics['L_mt']:.4f}")
    print(f"  L_unit: {metrics['L_unit']:.4f}")
    print(f"  Segment: {metrics['segment_length']}")
```

### Custom Loss Weights

```python
# Emphasize translation quality
criterion = EchoStreamMultiTaskCriterion(
    asr_weight=0.5,  # ← Reduce ASR weight
    st_weight=1.5,   # ← Increase ST weight
    mt_weight=2.0,   # ← Increase MT weight
    unit_weight=1.0,
)
```

### Offline Training (No Multi-chunk)

```python
# For offline S2ST (no latency constraint)
trainer = EchoStreamTrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    multi_chunk=False,  # ← Disable multi-chunk
)
```

---

## 🎯 Next Steps

✅ **Phase 3 완료!**

**다음 단계**:
- **Phase 4**: Alignment-based Policy (ASR/ST CTC 기반 READ/WRITE)
- **Phase 5**: Multi-chunk Training 통합 (Emformer + Multi-chunk)
- **Phase 6**: 전체 Agent 통합 및 테스트

---

## 📚 References

1. **StreamSpeech Paper**: "StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning"
2. **StreamSpeech Code**: `StreamSpeech_analysis/researches/ctc_unity/criterions/speech_to_speech_ctc_asr_st_criterion.py`
3. **CTC Loss**: Graves et al., "Connectionist Temporal Classification"
4. **Label Smoothing**: Szegedy et al., "Rethinking the Inception Architecture"

---

## 📝 Summary

| Component | Status | Description |
|-----------|--------|-------------|
| **Multi-task Criterion** | ✅ | 4개 loss 통합 (ASR, ST, MT, Unit) |
| **Multi-chunk Training** | ✅ | 랜덤 세그먼트 길이 학습 |
| **Label Smoothing** | ✅ | 일반화 성능 향상 |
| **Gradient Flow** | ✅ | 안정적인 학습 |
| **Test** | ✅ | 모든 테스트 통과 |

**Phase 3 완료! 🎉**

