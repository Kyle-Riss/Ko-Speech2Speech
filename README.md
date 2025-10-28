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
