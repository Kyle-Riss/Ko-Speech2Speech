#!/bin/bash
# EchoStream Training - Background Execution Script
# Mac M2 8GB RAM 환경에서 백그라운드 학습 실행

# 설정
CONFIG="configs/echostream_config.mac_m2_8gb.yaml"
TRAIN_MANIFEST="data/train_sampled.units.tsv"
DEV_MANIFEST="data/dev_sampled.units.tsv"
SAVE_DIR="checkpoints_mac_m2_8gb"
LOG_DIR="results"
NUM_WORKERS=0
EPOCHS=50

# 로그 파일명 (타임스탬프 포함)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_mac_m2_8gb_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_mac_m2_8gb_${TIMESTAMP}.pid"

# 로그 디렉토리 생성
mkdir -p "${LOG_DIR}"

# 학습 명령어
TRAIN_CMD="python scripts/train.py \
  --config ${CONFIG} \
  --train-manifest ${TRAIN_MANIFEST} \
  --dev-manifest ${DEV_MANIFEST} \
  --save-dir ${SAVE_DIR} \
  --num-workers ${NUM_WORKERS} \
  --epochs ${EPOCHS}"

echo "=========================================="
echo "EchoStream Training - Background Mode"
echo "=========================================="
echo "Config: ${CONFIG}"
echo "Save Dir: ${SAVE_DIR}"
echo "Log File: ${LOG_FILE}"
echo "PID File: ${PID_FILE}"
echo "=========================================="
echo ""

# 백그라운드 실행
nohup ${TRAIN_CMD} > "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!

# PID 저장
echo ${TRAIN_PID} > "${PID_FILE}"

echo "✅ Training started in background!"
echo "   PID: ${TRAIN_PID}"
echo "   Log: ${LOG_FILE}"
echo "   PID File: ${PID_FILE}"
echo ""
echo "To monitor progress:"
echo "   tail -f ${LOG_FILE}"
echo ""
echo "To check status:"
echo "   ps -p ${TRAIN_PID}"
echo ""
echo "To stop training:"
echo "   kill ${TRAIN_PID}"

