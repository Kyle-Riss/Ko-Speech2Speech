# 학습 재시작 권장 사항

**분석 날짜**: 2025-01-XX  
**상황**: Conformer → Emformer 인코더 교체 완료

---

## 🔍 현재 상황 분석

### 변경 사항
- ✅ **인코더**: Conformer → Emformer (구조 완전히 다름)
- ✅ **디코더**: 모두 동일 (ASR CTC, ST CTC, MT, Unit, Vocoder)
- ✅ **인터페이스**: StreamSpeech 형식 100% 유지

### 기존 체크포인트
- `checkpoints_mini_units_v4/checkpoint_best.pt` 존재
- `checkpoints_mini_units_v4/checkpoint_epoch_10.pt` 존재

---

## 💡 학습 전략

### ✅ 옵션 0: 기존 체크포인트 사용 (가장 권장!)

**현재 상황**: 이미 Emformer 기반 체크포인트가 존재합니다!

**확인 사항**:
1. 체크포인트가 최신 코드와 호환되는지 확인
2. 성능이 만족스러운지 확인
3. 필요시 추가 학습 (fine-tuning)

**사용 방법**:
```bash
# 추론 시 체크포인트 로드
python scripts/evaluate.py \
  --config configs/echostream_config.mini.yaml \
  --checkpoint checkpoints_mini_units_v4/checkpoint_best.pt
```

**추가 학습이 필요한 경우**:
- 성능이 부족한 경우
- 더 많은 데이터로 학습하고 싶은 경우
- 하이퍼파라미터 조정 후 재학습

---

### 옵션 1: 처음부터 재학습

**이유**:
1. **인코더 구조가 완전히 다름**
   - Conformer: Chunk-based attention, Depthwise conv
   - Emformer: Left Context Cache, Memory Bank
   - 가중치 호환 불가

2. **디코더는 동일하지만**
   - 인코더 출력이 달라질 수 있음
   - 처음부터 학습하는 것이 안정적

**장점**:
- ✅ 깨끗한 학습 (호환성 문제 없음)
- ✅ Emformer 특성에 최적화된 학습
- ✅ 안정적인 수렴

**단점**:
- ❌ 시간이 오래 걸림
- ❌ 기존 학습 결과 활용 불가

---

### 옵션 2: 부분 로딩 (디코더만 재사용)

**전략**:
1. 기존 체크포인트에서 디코더 가중치만 추출
2. 인코더는 랜덤 초기화
3. 디코더는 기존 가중치로 초기화
4. 전체 모델 학습 (인코더 학습률 높게, 디코더 학습률 낮게)

**장점**:
- ✅ 디코더 가중치 재사용 가능
- ✅ 학습 시간 단축 가능

**단점**:
- ⚠️ 구현 복잡도 증가
- ⚠️ 인코더-디코더 불일치 가능성

**구현 예시**:
```python
# 기존 체크포인트 로드
old_checkpoint = torch.load('checkpoints_mini_units_v4/checkpoint_best.pt')
old_state = old_checkpoint['model']

# 새 모델 생성
new_model = build_echostream_model(config)

# 디코더 가중치만 복사
decoder_keys = [
    'asr_ctc_decoder', 'st_ctc_decoder', 
    'mt_decoder', 'unit_decoder', 'vocoder'
]

for key in decoder_keys:
    for param_name, param_value in old_state.items():
        if param_name.startswith(f'{key}.'):
            new_model.state_dict()[param_name].copy_(param_value)

# 인코더는 랜덤 초기화 (이미 새로 생성됨)
```

---

### 옵션 3: Transfer Learning (인코더만 학습)

**전략**:
1. 디코더는 기존 가중치로 고정 (frozen)
2. 인코더만 학습
3. 이후 전체 fine-tuning

**장점**:
- ✅ 디코더 가중치 보존
- ✅ 인코더에 집중 학습

**단점**:
- ⚠️ 인코더-디코더 불일치 가능
- ⚠️ 성능 저하 가능

---

## ✅ 권장 사항

### 상황별 권장

#### 0. 기존 체크포인트 확인 (최우선!)
**→ 먼저 기존 체크포인트로 추론 테스트**

```bash
# 1. 기존 체크포인트로 추론 테스트
python scripts/evaluate.py \
  --config configs/echostream_config.mini.yaml \
  --checkpoint checkpoints_mini_units_v4/checkpoint_best.pt \
  --test-manifest data/test_sampled.tsv

# 2. 성능 확인
# - 만족스러우면 → 추가 학습 불필요!
# - 개선 필요하면 → 아래 옵션 선택
```

#### 1. 시간이 충분하고 성능 개선이 필요한 경우
**→ 옵션 1: 처음부터 재학습**

```bash
# 기존 체크포인트 백업
mv checkpoints_mini_units_v4 checkpoints_mini_units_v4_backup

# 새로 학습 시작
python scripts/train.py \
  --config configs/echostream_config.mini.yaml \
  --train-manifest data/train_sampled.units.tsv \
  --dev-manifest data/dev_sampled.units.tsv \
  --save-dir checkpoints_mini_units_v5 \
  --num-workers 0
```

#### 2. 시간이 부족한 경우
**→ 옵션 2: 부분 로딩 (디코더 재사용)**

학습 스크립트에 부분 로딩 기능 추가 필요.

---

## 🔧 학습 스크립트 수정 (부분 로딩 지원)

현재 `scripts/train.py`에는 체크포인트 로딩 기능이 없습니다. 
부분 로딩을 원한다면 다음 기능을 추가해야 합니다:

```python
# scripts/train.py에 추가할 코드

def load_partial_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    load_encoder: bool = False,
    load_decoders: bool = True,
):
    """
    부분 체크포인트 로딩.
    
    Args:
        model: 새로 생성된 모델
        checkpoint_path: 기존 체크포인트 경로
        load_encoder: 인코더 가중치 로드 여부 (False 권장)
        load_decoders: 디코더 가중치 로드 여부 (True 권장)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    old_state = checkpoint.get('model', checkpoint)
    new_state = model.state_dict()
    
    loaded_keys = []
    skipped_keys = []
    
    for key, value in old_state.items():
        # 인코더 스킵 (구조가 다름)
        if key.startswith('encoder.') and not load_encoder:
            skipped_keys.append(key)
            continue
        
        # 디코더만 로드
        if load_decoders and any(key.startswith(prefix) for prefix in [
            'asr_ctc_decoder.',
            'st_ctc_decoder.',
            'mt_decoder.',
            'unit_decoder.',
            'vocoder.',
        ]):
            if key in new_state and new_state[key].shape == value.shape:
                new_state[key].copy_(value)
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key} (shape mismatch)")
        else:
            skipped_keys.append(key)
    
    model.load_state_dict(new_state, strict=False)
    
    logger.info(f"Loaded {len(loaded_keys)} keys from checkpoint")
    logger.info(f"Skipped {len(skipped_keys)} keys")
    
    return loaded_keys, skipped_keys
```

---

## 📊 체크포인트 호환성 확인

### 현재 체크포인트 구조
```python
checkpoint = {
    'epoch': int,
    'model': {...},  # 모델 state_dict
    'optimizer': {...},
    'loss': float,
}
```

### 호환성 체크
1. **인코더**: ❌ 호환 불가 (Conformer → Emformer)
2. **디코더**: ✅ 호환 가능 (구조 동일)
3. **Vocoder**: ✅ 호환 가능 (구조 동일)

---

## 🎯 최종 권장 사항

### 1단계: 기존 체크포인트 테스트 (최우선!)

**먼저 확인**:
```bash
# 기존 체크포인트로 추론 테스트
python scripts/evaluate.py \
  --config configs/echostream_config.mini.yaml \
  --checkpoint checkpoints_mini_units_v4/checkpoint_best.pt
```

**결과에 따라**:
- ✅ 성능이 만족스러우면 → **추가 학습 불필요!**
- ⚠️ 성능 개선이 필요하면 → 아래 옵션 선택

---

### 2단계: 추가 학습이 필요한 경우

#### 추천: 기존 체크포인트에서 이어서 학습 (Resume)

**이유**:
1. **이미 Emformer 기반**: 기존 체크포인트가 Emformer 구조
2. **시간 절약**: 처음부터보다 빠름
3. **안정성**: 기존 학습 결과 활용

**구현 필요**: `train.py`에 resume 기능 추가

#### 대안: 처음부터 재학습

**이유**:
1. **구조적 차이**: Conformer와 Emformer는 완전히 다른 구조
2. **안정성**: 처음부터 학습하는 것이 가장 안정적
3. **최적화**: Emformer 특성에 맞게 최적화된 학습

**실행 방법**:
```bash
# 1. 기존 체크포인트 백업
mkdir -p checkpoints_backup
cp -r checkpoints_mini_units_v4 checkpoints_backup/

# 2. 새 학습 시작
python scripts/train.py \
  --config configs/echostream_config.mini.yaml \
  --train-manifest data/train_sampled.units.tsv \
  --dev-manifest data/dev_sampled.units.tsv \
  --save-dir checkpoints_mini_units_v5 \
  --num-workers 0 \
  --batch-size 8
```

### 대안: 부분 로딩 (고급)

시간이 부족하고 디코더 가중치를 재사용하고 싶다면:
1. 위의 `load_partial_checkpoint` 함수를 `train.py`에 추가
2. 학습 시작 전에 디코더만 로드
3. 인코더는 랜덤 초기화로 시작

---

## ⚠️ 주의사항

1. **인코더 가중치 재사용 불가**
   - Conformer와 Emformer는 구조가 완전히 다름
   - 강제로 로드하면 오류 발생

2. **디코더 가중치 재사용 시**
   - 인코더 출력 분포가 달라질 수 있음
   - Fine-tuning이 필요할 수 있음

3. **학습률 조정**
   - 부분 로딩 시: 인코더 학습률 높게, 디코더 학습률 낮게
   - 처음부터 학습 시: 동일한 학습률 사용

---

## 📝 체크리스트

학습 시작 전 확인:

- [ ] 기존 체크포인트 백업 완료
- [ ] Config 파일 확인 (`echostream_config.mini.yaml`)
- [ ] 데이터 경로 확인 (train/dev manifests)
- [ ] Units 파일 존재 확인 (`data/units/`)
- [ ] Vocoder 체크포인트 경로 확인
- [ ] GPU/CPU 설정 확인
- [ ] 학습 스크립트 실행 권한 확인

---

**결론**: **처음부터 재학습을 권장합니다.** 구조가 완전히 다르므로 안정적이고 깨끗한 학습이 가능합니다.

---

**마지막 업데이트**: 2025-01-XX  
**버전**: 1.0

