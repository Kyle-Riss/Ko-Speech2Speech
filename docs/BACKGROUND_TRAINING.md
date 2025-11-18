# 백그라운드 학습 가이드

## 🔍 현재 상태 확인

현재 학습 프로세스는 **포그라운드**로 실행 중입니다. 터미널을 닫으면 학습이 중단됩니다.

## 🚀 백그라운드로 실행하기

### 방법 1: 스크립트 사용 (권장)

```bash
# 백그라운드로 학습 시작
./scripts/run_training_background.sh
```

이 스크립트는:
- ✅ `nohup`으로 백그라운드 실행
- ✅ 로그 파일 자동 생성 (`results/train_mac_m2_8gb_TIMESTAMP.log`)
- ✅ PID 파일 저장 (`results/train_mac_m2_8gb_TIMESTAMP.pid`)
- ✅ 터미널 종료 후에도 계속 실행

### 방법 2: 수동 실행

```bash
# 백그라운드로 실행
nohup python scripts/train.py \
  --config configs/echostream_config.mac_m2_8gb.yaml \
  --train-manifest data/train_sampled.units.tsv \
  --dev-manifest data/dev_sampled.units.tsv \
  --save-dir checkpoints_mac_m2_8gb \
  --num-workers 0 \
  --epochs 50 \
  > results/train_background.log 2>&1 &

# PID 확인
echo $!
```

## 📊 학습 상태 확인

### 스크립트 사용

```bash
# 학습 상태 확인
./scripts/check_training.sh
```

이 스크립트는 다음을 확인합니다:
- ✅ 프로세스 실행 여부
- ✅ CPU/메모리 사용률
- ✅ 실행 시간
- ✅ 최신 로그 파일
- ✅ 체크포인트 파일

### 수동 확인

```bash
# 프로세스 확인
ps aux | grep "python.*train.py" | grep -v grep

# 로그 실시간 확인
tail -f results/train_mac_m2_8gb_*.log

# 최신 로그 확인
ls -lt results/train_mac_m2_8gb_*.log | head -1 | awk '{print $NF}' | xargs tail -f
```

## 🛑 학습 중지

### PID 파일이 있는 경우

```bash
# PID 확인
cat results/train_mac_m2_8gb_*.pid

# 프로세스 종료
kill $(cat results/train_mac_m2_8gb_*.pid)
```

### PID 파일이 없는 경우

```bash
# 프로세스 찾기
ps aux | grep "python.*train.py" | grep -v grep

# 프로세스 종료 (PID는 위 명령어 결과에서 확인)
kill <PID>
```

## 📝 로그 파일 위치

백그라운드 실행 시 로그는 다음 위치에 저장됩니다:
- `results/train_mac_m2_8gb_YYYYMMDD_HHMMSS.log`

로그 파일에는 다음이 포함됩니다:
- 학습 진행 상황
- Loss 값
- 에러 메시지
- 체크포인트 저장 정보

## 💡 팁

### 1. 여러 학습 동시 실행

다른 설정으로 여러 학습을 동시에 실행하려면:
```bash
# 각각 다른 로그 파일로 실행
nohup python scripts/train.py --config config1.yaml ... > log1.log 2>&1 &
nohup python scripts/train.py --config config2.yaml ... > log2.log 2>&1 &
```

### 2. 로그 실시간 모니터링

```bash
# 여러 로그 파일 동시 모니터링
tail -f results/train_*.log
```

### 3. 학습 완료 알림

학습이 완료되면 체크포인트 파일이 생성됩니다:
```bash
# 체크포인트 확인
ls -lh checkpoints_mac_m2_8gb/*.pt
```

## ⚠️ 주의사항

1. **터미널 종료**: 포그라운드 실행 시 터미널을 닫으면 학습이 중단됩니다
2. **백그라운드 권장**: 장시간 학습은 반드시 백그라운드로 실행하세요
3. **로그 확인**: 정기적으로 로그를 확인하여 에러가 없는지 확인하세요
4. **디스크 공간**: 체크포인트 파일이 커질 수 있으므로 디스크 공간을 확인하세요

---

**마지막 업데이트**: 2025-11-18

