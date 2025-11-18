#!/bin/bash
# EchoStream Training Status Checker

LOG_DIR="results"
SAVE_DIR="checkpoints_mac_m2_8gb"

echo "=========================================="
echo "EchoStream Training Status"
echo "=========================================="
echo ""

# PID 파일 확인
PID_FILES=$(find "${LOG_DIR}" -name "train_mac_m2_8gb_*.pid" 2>/dev/null | sort -r | head -1)

if [ -z "${PID_FILES}" ]; then
    echo "❌ No training PID files found"
else
    LATEST_PID_FILE=$(echo "${PID_FILES}" | head -1)
    PID=$(cat "${LATEST_PID_FILE}" 2>/dev/null)
    
    if [ -z "${PID}" ]; then
        echo "❌ PID file exists but is empty"
    else
        # 프로세스 확인
        if ps -p "${PID}" > /dev/null 2>&1; then
            echo "✅ Training is RUNNING"
            echo "   PID: ${PID}"
            echo "   PID File: ${LATEST_PID_FILE}"
            
            # CPU 사용률 확인
            CPU_USAGE=$(ps -p "${PID}" -o %cpu= | tr -d ' ')
            MEM_USAGE=$(ps -p "${PID}" -o %mem= | tr -d ' ')
            echo "   CPU: ${CPU_USAGE}%"
            echo "   Memory: ${MEM_USAGE}%"
            
            # 실행 시간 확인
            ELAPSED=$(ps -p "${PID}" -o etime= | tr -d ' ')
            echo "   Elapsed: ${ELAPSED}"
        else
            echo "❌ Training is NOT running (PID ${PID} not found)"
            echo "   PID File: ${LATEST_PID_FILE}"
        fi
    fi
fi

echo ""

# 로그 파일 확인
LOG_FILES=$(find "${LOG_DIR}" -name "train_mac_m2_8gb_*.log" 2>/dev/null | sort -r | head -1)

if [ -n "${LOG_FILES}" ]; then
    LATEST_LOG=$(echo "${LOG_FILES}" | head -1)
    echo "📄 Latest Log: ${LATEST_LOG}"
    echo ""
    echo "Last 10 lines:"
    echo "----------------------------------------"
    tail -10 "${LATEST_LOG}" 2>/dev/null || echo "Cannot read log file"
    echo "----------------------------------------"
fi

echo ""

# 체크포인트 확인
if [ -d "${SAVE_DIR}" ]; then
    CHECKPOINTS=$(find "${SAVE_DIR}" -name "*.pt" 2>/dev/null | wc -l | tr -d ' ')
    if [ "${CHECKPOINTS}" -gt 0 ]; then
        echo "✅ Checkpoints found: ${CHECKPOINTS} file(s)"
        LATEST_CKPT=$(find "${SAVE_DIR}" -name "*.pt" -type f -exec ls -t {} + | head -1)
        if [ -n "${LATEST_CKPT}" ]; then
            CKPT_SIZE=$(ls -lh "${LATEST_CKPT}" | awk '{print $5}')
            CKPT_TIME=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "${LATEST_CKPT}" 2>/dev/null || stat -c "%y" "${LATEST_CKPT}" 2>/dev/null | cut -d'.' -f1)
            echo "   Latest: $(basename ${LATEST_CKPT}) (${CKPT_SIZE}, ${CKPT_TIME})"
        fi
    else
        echo "⚠️  No checkpoints found yet"
    fi
else
    echo "⚠️  Checkpoint directory not found: ${SAVE_DIR}"
fi

echo ""
echo "=========================================="

