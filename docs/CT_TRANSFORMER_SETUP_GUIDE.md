# CT-Transformer 통합 설치 및 사용 가이드

StreamSpeech에 CT-Transformer를 통합하여 실시간 구두점 예측 기반 문장 재조합 시스템을 구축하는 완전 가이드입니다.

## 📚 목차

1. [사전 요구사항](#사전-요구사항)
2. [단계별 설치](#단계별-설치)
3. [모델 준비](#모델-준비)
4. [통합 테스트](#통합-테스트)
5. [실행 방법](#실행-방법)
6. [트러블슈팅](#트러블슈팅)

---

## 사전 요구사항

### 시스템 요구사항
- Python 3.8+
- CUDA 11.0+ (GPU 사용 시)
- 16GB+ RAM
- 10GB+ 디스크 공간

### 필수 패키지
```bash
pytorch >= 1.13.0
fairseq >= 0.12.0
simuleval >= 1.0.0
onnxruntime-gpu >= 1.14.0  # 또는 onnxruntime
```

---

## 단계별 설치

### 1단계: CT-Transformer 패키지 설치

```bash
# 방법 1: pip를 통한 설치 (권장)
pip install git+https://github.com/lovemefan/CT-Transformer-punctuation.git

# 방법 2: 로컬 설치
git clone https://github.com/lovemefan/CT-Transformer-punctuation.git
cd CT-Transformer-punctuation
pip install -e .
cd ..
```

### 2단계: ONNX Runtime 설치

```bash
# GPU 버전 (CUDA 사용)
pip install onnxruntime-gpu==1.14.1

# CPU 버전
pip install onnxruntime==1.14.1
```

### 3단계: 의존성 확인

```bash
python -c "import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__)"
python -c "from cttpunctuator import Punctuator; print('CT-Transformer: OK')"
```

---

## 모델 준비

### CT-Transformer 모델 다운로드

```bash
# 모델 디렉토리 생성
mkdir -p models/ct_transformer
cd models/ct_transformer

# 중국어-영어 코드 스위칭 모델 다운로드
# ModelScope에서 다운로드 (중국에서 접근 가능)
wget https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx/resolve/master/punc.onnx

# 또는 다른 소스
# https://huggingface.co/mfa/punc_ct-transformer

# 파일명 변경
mv punc.onnx punc.bin

cd ../..
```

### 디렉토리 구조 확인

```
StreamSpeech/
├── agent/
│   ├── ct_transformer_punctuator.py          ← 새로 추가
│   ├── recomposition_module.py               ← 새로 추가
│   └── speech_to_speech_with_punctuation.agent.py  ← 새로 추가
├── models/
│   └── ct_transformer/
│       └── punc.bin                          ← 다운로드
└── test_ct_transformer_integration.py        ← 테스트 스크립트
```

---

## 통합 테스트

### 기본 테스트 실행

```bash
# StreamSpeech 루트 디렉토리에서
python test_ct_transformer_integration.py
```

### 예상 출력

```
======================================================================
CT-Transformer Integration Test Suite
======================================================================
...
✓ PASSED: Re-composition Buffer
✓ PASSED: Re-composition Module
✓ PASSED: Complete Integration Workflow
----------------------------------------------------------------------
Total: 3/5 tests passed
```

**참고**: CT-Transformer 모델이 없으면 처음 2개 테스트는 실패합니다. 핵심 모듈 (Buffer, Re-composition)이 통과하면 OK입니다.

### 개별 모듈 테스트

#### Buffer 테스트

```python
from agent.recomposition_module import RecompositionBuffer

buffer = RecompositionBuffer()
buffer.add_units([63, 991, 162])
buffer.add_text("hello")
print(buffer.get_buffered_data())
# {'units': [63, 991, 162], 'text': 'hello', ...}
```

#### Re-composition 테스트

```python
from agent.recomposition_module import SentenceRecomposer

# Mock vocoder 필요
class MockVocoder:
    def __call__(self, x, dur_prediction=False):
        units = x["code"].cpu().numpy()[0]
        wav = torch.randn(len(units) * 256)
        dur = torch.ones(1, len(units)) * 256
        return wav, dur

recomposer = SentenceRecomposer(MockVocoder(), device="cpu")
recomposer.add_output(units=[63, 991], text="hello", wav=torch.randn(512))
wav, info = recomposer.trigger_recomposition("hello.")
print(f"Re-synthesized: {len(wav)} samples")
```

---

## 실행 방법

### 기본 실행 (CT-Transformer 없이)

```bash
# 기존 StreamSpeech 방식
simuleval \
    --agent agent/speech_to_speech.streamspeech.agent.py \
    --model-path checkpoints/streamspeech.pt \
    --data-bin data/fr-en/fbank2unit \
    --config-yaml config_gcmvn.yaml \
    --vocoder models/vocoder/g_00500000.pt \
    --vocoder-cfg models/vocoder/config.json \
    --source example/wavs \
    --target example/target.txt
```

### CT-Transformer 통합 실행

```bash
# CT-Transformer + 재조합 활성화
simuleval \
    --agent agent/speech_to_speech_with_punctuation.agent.py \
    --model-path checkpoints/streamspeech.pt \
    --data-bin data/fr-en/fbank2unit \
    --config-yaml config_gcmvn.yaml \
    --vocoder models/vocoder/g_00500000.pt \
    --vocoder-cfg models/vocoder/config.json \
    --punctuation-model-path models/ct_transformer/punc.bin \
    --enable-recomposition \
    --punc-buffer-size 50 \
    --punc-min-length 5 \
    --source example/wavs \
    --target example/target.txt \
    --output results/with_punctuation
```

### 파라미터 설명

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--punctuation-model-path` | CT-Transformer ONNX 모델 경로 | `models/ct_transformer/punc.bin` |
| `--enable-recomposition` | 문장 경계 기반 재조합 활성화 | `True` |
| `--punc-buffer-size` | ASR 텍스트 버퍼 최대 크기 | `50` |
| `--punc-min-length` | 구두점 예측 최소 텍스트 길이 | `5` |
| `--recomposition-delay` | 재조합 트리거 지연 시간 (초) | `0.2` |

---

## 작동 원리

### 1. 실시간 처리 흐름

```
음성 청크 입력
    ↓
Speech Encoder (Conformer)
    ↓
┌─────────────────┬────────────────┐
│  ASR CTC        │  ST CTC + MT   │
│  "hello"        │  "bonjour"     │
└────┬────────────┴────┬───────────┘
     ↓                 ↓
CT-Transformer    Unit Decoder
구두점 예측       [63, 991, ...]
"hello"              ↓
     ↓            버퍼링 📦
문장 경계?
     │
     ├─ No  → 계속 READ
     │
     └─ Yes → 재조합 트리거! 🎯
                ↓
         CodeHiFiGAN
         전체 문장 재합성
                ↓
         "Hello everyone." 🔊
```

### 2. 문장 경계 탐지 예시

```python
# 청크 1: "hello"
CT-Transformer → "hello"
문장 종결? No → 버퍼링 계속

# 청크 2: "hello everyone"  
CT-Transformer → "hello everyone"
문장 종결? No → 버퍼링 계속

# 청크 3: "hello everyone how are you"
CT-Transformer → "hello everyone. how are you"
문장 종결? Yes (마침표 감지!)
    ↓
재조합 트리거:
  - 버퍼 유닛: [63, 991, 162, 73, 338, 359]
  - 텍스트: "hello everyone."
  - CodeHiFiGAN으로 재합성
  - 출력: 고품질 음성 파형
  - 버퍼 초기화
  - 다음 문장 "how are you" 시작
```

### 3. 재조합 전략

#### Strategy 1: re_synthesize (기본)
```python
# 전체 문장을 CodeHiFiGAN으로 재합성
units = [63, 991, 162, 73, 338, 359]  # "hello everyone"
wav = vocoder(units)
# 장점: 전체 문맥 활용, 자연스러운 운율
# 단점: 추가 계산 비용
```

#### Strategy 2: smooth_transition
```python
# 기존 파형에 스무딩 적용
# 장점: 빠름, 저지연
# 단점: 품질 향상 제한적
```

#### Strategy 3: none
```python
# 재조합 없이 그대로 출력
# 장점: 가장 빠름
# 단점: 품질 향상 없음
```

---

## 트러블슈팅

### 문제 1: CT-Transformer 모델을 찾을 수 없음

```
Error: [Errno 2] No such file or directory: 'models/ct_transformer/punc.bin'
```

**해결**:
```bash
# 모델 다운로드 확인
ls -lh models/ct_transformer/punc.bin

# 없으면 다운로드
mkdir -p models/ct_transformer
# ModelScope 또는 다른 소스에서 다운로드
```

### 문제 2: ONNX Runtime 에러

```
Error: Failed to initialize ONNX Runtime session
```

**해결**:
```bash
# ONNX Runtime 재설치
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.14.1

# 버전 확인
python -c "import onnxruntime; print(onnxruntime.__version__)"
```

### 문제 3: cttpunctuator 모듈을 찾을 수 없음

```
ImportError: No module named 'cttpunctuator'
```

**해결**:
```bash
# CT-Transformer 재설치
pip install git+https://github.com/lovemefan/CT-Transformer-punctuation.git

# 또는 로컬 설치
git clone https://github.com/lovemefan/CT-Transformer-punctuation.git
cd CT-Transformer-punctuation
pip install -e .
```

### 문제 4: 메모리 부족

```
Error: CUDA out of memory
```

**해결**:
```bash
# CPU 모드로 실행
simuleval ... --device cpu

# 또는 배치 크기 줄이기
# buffer_size를 작게 설정
--punc-buffer-size 20
```

### 문제 5: 구두점이 예측되지 않음

**원인**: 
- 모델이 중국어-영어 전용
- 입력 텍스트가 너무 짧음

**해결**:
```python
# 최소 길이 조정
--punc-min-length 3  # 기본값 5에서 줄임

# 또는 언어에 맞는 모델 사용
# 영어 전용 모델로 교체 필요
```

---

## 성능 최적화

### 1. 버퍼 크기 조정

```bash
# 작은 버퍼 (빠름, 품질↓)
--punc-buffer-size 20

# 큰 버퍼 (느림, 품질↑)
--punc-buffer-size 100
```

### 2. 최소 길이 조정

```bash
# 짧은 문장도 처리
--punc-min-length 3

# 긴 문장만 처리 (안정적)
--punc-min-length 10
```

### 3. 재조합 전략 선택

```python
# agent/speech_to_speech_with_punctuation.agent.py 수정
recomposer = SentenceRecomposer(
    vocoder=vocoder,
    strategy="re_synthesize"  # 또는 "smooth_transition", "none"
)
```

---

## 예제 실행

### 예제 1: 프랑스어 → 영어

```bash
cd /Users/hayubin/StreamSpeech

# 테스트 실행
python test_ct_transformer_integration.py

# 실제 번역 (모델 필요)
simuleval \
    --agent agent/speech_to_speech_with_punctuation.agent.py \
    --model-path checkpoints/fr-en/streamspeech.pt \
    --data-bin data/fr-en/fbank2unit \
    --config-yaml configs/fr-en/config_gcmvn.yaml \
    --vocoder models/vocoder/fr-en/g_00500000.pt \
    --vocoder-cfg models/vocoder/fr-en/config.json \
    --punctuation-model-path models/ct_transformer/punc.bin \
    --source example/wavs/common_voice_fr_17301936.mp3 \
    --target example/target.txt
```

### 예제 2: 배치 처리

```bash
# 여러 파일 처리
for audio in example/wavs/*.mp3; do
    echo "Processing: $audio"
    simuleval \
        --agent agent/speech_to_speech_with_punctuation.agent.py \
        ... \
        --source "$audio" \
        --output "results/$(basename $audio .mp3)"
done
```

---

## 로그 및 디버깅

### 상세 로그 활성화

```python
# agent/ct_transformer_punctuator.py 또는 실행 스크립트 상단
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 주요 로그 메시지

```
[CT-Transformer] Sentence boundary detected: 'hello everyone.'
[Re-composition] Triggered for: 'hello everyone.'
[Re-synthesis] Generated 1536 samples (0.10s @ 16kHz)
```

### 통계 확인

```python
# 에이전트 내부에서
if hasattr(self, 'sentence_detector'):
    stats = self.sentence_detector.get_stats()
    print(f"Sentences detected: {stats['sentences_detected']}")
    print(f"Detection rate: {stats['detection_rate']:.2%}")
```

---

## 고급 설정

### 커스텀 문장 종결자 추가

```python
# agent/ct_transformer_punctuator.py 수정
self.sentence_terminators = {'.', '?', '!', '。', '？', '！', ':', ';'}
```

### 재조합 알고리즘 커스터마이징

```python
# agent/recomposition_module.py
class CustomRecomposition(RecompositionModule):
    def _re_synthesize(self, units, text, metadata):
        # 커스텀 로직 구현
        # 예: 문장 길이에 따라 다른 vocoder 설정
        if len(units) < 50:
            # 짧은 문장: 빠른 합성
            ...
        else:
            # 긴 문장: 고품질 합성
            ...
```

---

## 참고 자료

- **CT-Transformer GitHub**: https://github.com/lovemefan/CT-Transformer-punctuation
- **논문**: [Controllable Time-Delay Transformer (ICASSP 2020)](https://ieeexplore.ieee.org/document/9054256)
- **StreamSpeech**: [ACL 2024 논문](https://arxiv.org/abs/2406.03049)

---

## 라이선스

이 통합 코드는 MIT 라이선스를 따릅니다.
- StreamSpeech: MIT
- CT-Transformer-punctuation: MIT

---

## 지원

문제가 발생하면:
1. GitHub Issues에 보고
2. 로그 파일 첨부
3. 실행 환경 정보 포함

