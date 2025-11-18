# Mac M2 8GB RAM 파인튜닝 가이드

**환경**: Mac M2, RAM 8GB  
**목적**: 메모리 제약 환경에서 안정적인 파인튜닝

---

## 🎯 핵심 전략

### 1. 작은 모델 크기
- Encoder: 4 layers (기존 16 → 4)
- Embed dim: 128 (기존 256 → 128)
- Attention heads: 2 (기존 4 → 2)
- FFN dim: 512 (기존 1024 → 512)

### 2. 작은 배치 사이즈 + Gradient Accumulation
- Batch size: 2 (매우 작음)
- Update freq: 8 (gradient accumulation)
- **Effective batch size**: 2 × 8 = 16

### 3. 메모리 최적화 설정
- `num_workers: 0` (Mac multiprocessing 이슈 방지)
- `pin_memory: false` (CPU 사용 시 불필요)
- `fp16: false` (Mac에서 안정성)

---

## 📋 설정 파일

### Mac M2 8GB 전용 Config

**파일**: `configs/echostream_config.mac_m2_8gb.yaml`

주요 설정:
```yaml
encoder:
  embed_dim: 128      # 작은 임베딩
  layers: 4           # 적은 레이어
  attention_heads: 2
  ffn_embed_dim: 512

training:
  batch_size: 2       # 매우 작은 배치
  update_freq: 8      # Gradient accumulation
  max_tokens: 5000    # 작은 토큰 수

hardware:
  num_workers: 0      # Mac multiprocessing 방지
  fp16: false         # 안정성
  pin_memory: false
```

---

## 🚀 파인튜닝 실행 방법

### 1. 기존 체크포인트에서 이어서 학습 (Resume)

**현재 상황**: 이미 Emformer 기반 체크포인트가 있음
- `checkpoints_mini_units_v4/checkpoint_best.pt` (Epoch 9)

**추가 학습 방법**:

```bash
# Mac M2 8GB 최적화 설정으로 파인튜닝
python scripts/train.py \
  --config configs/echostream_config.mac_m2_8gb.yaml \
  --train-manifest data/train_sampled.units.tsv \
  --dev-manifest data/dev_sampled.units.tsv \
  --save-dir checkpoints_mini_units_v5 \
  --num-workers 0 \
  --batch-size 2 \
  --epochs 50
```

**주의사항**:
- 현재 `train.py`에는 resume 기능이 없음
- 기존 체크포인트를 로드하려면 코드 수정 필요

---

### 2. 처음부터 학습 (권장)

**이유**: Mac M2 8GB에서는 작은 모델로 처음부터 학습하는 것이 안정적

```bash
# Mac M2 8GB 최적화 설정
python scripts/train.py \
  --config configs/echostream_config.mac_m2_8gb.yaml \
  --train-manifest data/train_sampled.units.tsv \
  --dev-manifest data/dev_sampled.units.tsv \
  --save-dir checkpoints_mac_m2_8gb \
  --num-workers 0 \
  --epochs 50
```

---

## 💾 메모리 사용량 예상

### 모델 크기 (Mac M2 설정)

```
Encoder (4L, 128d):      ~2.5M parameters
ASR CTC:                 ~0.8M
ST CTC (1L):             ~0.3M
MT Decoder (2L):         ~1.0M
Unit Decoder (2L):       ~1.2M
Total:                   ~5.8M parameters (~22 MB @ fp32)
```

### 학습 시 메모리

```
모델:            ~22 MB
Gradients:       ~22 MB
Optimizer state: ~44 MB (Adam)
Batch (size=2):  ~50 MB
Total:           ~138 MB (안전!)
```

**8GB RAM에서 충분히 여유 있음!**

---

## ⚙️ Gradient Accumulation 동작

### 작동 원리

```python
# update_freq = 8일 때
for batch in dataloader:
    loss = model(batch) / 8  # Loss 스케일링
    loss.backward()          # Gradient 누적
    
    if accumulated_batches >= 8:
        optimizer.step()    # 8개 배치 누적 후 업데이트
        optimizer.zero_grad()
```

### 효과

- **메모리**: 배치 사이즈 2만 사용 (메모리 절약)
- **학습 효과**: Effective batch size = 16 (큰 배치와 동일)
- **안정성**: 작은 배치로 메모리 오버플로우 방지

---

## 🔧 추가 최적화 팁

### 1. 데이터 로딩 최적화

```yaml
# configs/echostream_config.mac_m2_8gb.yaml
hardware:
  num_workers: 0  # Mac에서 필수 (multiprocessing 이슈)
```

### 2. 짧은 발화만 사용

```yaml
data:
  max_duration: 10.0  # 10초 이하만 사용 (메모리 절약)
```

### 3. 체크포인트 저장 간격 조정

```bash
# 자주 저장하지 않기 (디스크 I/O 감소)
--save-interval 20  # 20 epoch마다 저장
```

### 4. Mixed Precision 비활성화

```yaml
hardware:
  fp16: false  # Mac M2에서 안정성 우선
```

---

## 📊 예상 학습 시간

### Mac M2 8GB 환경

```
모델 크기: 5.8M parameters
배치 사이즈: 2
Update freq: 8
Effective batch: 16

예상 속도:
- 1 epoch: ~30-60분 (데이터 크기에 따라)
- 50 epochs: ~25-50시간
```

**권장**: 밤새 학습 또는 백그라운드 실행

---

## ✅ 체크리스트

학습 시작 전:

- [x] soundfile 설치 완료
- [ ] Config 파일 확인 (`echostream_config.mac_m2_8gb.yaml`)
- [ ] 데이터 경로 확인
- [ ] Units 파일 존재 확인
- [ ] 메모리 여유 공간 확인 (최소 2GB)
- [ ] 배치 사이즈 확인 (2)
- [ ] Update freq 확인 (8)
- [ ] num_workers 확인 (0)

---

## 🚨 문제 해결

### 메모리 부족 오류

```
RuntimeError: CUDA out of memory
→ Mac M2는 CPU 사용, 이 오류는 발생하지 않음

OSError: [Errno 12] Cannot allocate memory
→ 배치 사이즈를 1로 줄이기
→ update_freq를 16으로 늘리기
```

### 해결 방법

```yaml
# 더 작은 설정
training:
  batch_size: 1       # 2 → 1
  update_freq: 16     # 8 → 16 (effective batch 유지)
  max_tokens: 3000    # 5000 → 3000
```

---

## 🎯 최종 권장 사항

### Mac M2 8GB 환경

1. **작은 모델 사용**: `echostream_config.mac_m2_8gb.yaml`
2. **작은 배치 + Accumulation**: batch_size=2, update_freq=8
3. **CPU 사용**: GPU 없이도 충분히 빠름 (M2 Neural Engine 활용)
4. **안정성 우선**: fp16 비활성화, num_workers=0

### 실행 명령어

```bash
# 최종 권장 명령어
python scripts/train.py \
  --config configs/echostream_config.mac_m2_8gb.yaml \
  --train-manifest data/train_sampled.units.tsv \
  --dev-manifest data/dev_sampled.units.tsv \
  --save-dir checkpoints_mac_m2_8gb \
  --num-workers 0 \
  --epochs 50
```

---

**준비 완료!** Mac M2 8GB 환경에서 안정적으로 파인튜닝할 수 있습니다! 🚀

---

**마지막 업데이트**: 2025-01-XX  
**버전**: 1.0

