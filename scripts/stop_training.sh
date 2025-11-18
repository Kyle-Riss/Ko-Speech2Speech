#!/bin/bash
# EchoStream Training Stopper

LOG_DIR="results"

echo "=========================================="
echo "EchoStream Training Stopper"
echo "=========================================="
echo ""

# PID 파일 찾기
PID_FILES=$(find "${LOG_DIR}" -name "train_mac_m2_8gb_*.pid" 2>/dev/null | sort -r)

if [ -z "${PID_FILES}" ]; then
    echo "❌ No training PID files found"
    echo ""
    echo "Trying to find training processes manually..."
    TRAIN_PIDS=$(ps aux | grep "python.*train.py" | grep -v grep | awk '{print $2}')
    if [ -z "${TRAIN_PIDS}" ]; then
        echo "✅ No training processes found - all stopped"
        exit 0
    else
        echo "Found training processes:"
        for PID in ${TRAIN_PIDS}; do
            CMD=$(ps -p ${PID} -o command= 2>/dev/null)
            CPU=$(ps -p ${PID} -o %cpu= 2>/dev/null | tr -d ' ')
            MEM=$(ps -p ${PID} -o %mem= 2>/dev/null | tr -d ' ')
            ETIME=$(ps -p ${PID} -o etime= 2>/dev/null | tr -d ' ')
            echo "  PID ${PID}: CPU ${CPU}%, MEM ${MEM}%, Elapsed ${ETIME}"
            echo "    Command: ${CMD}"
        done
        echo ""
        read -p "Kill these processes? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            for PID in ${TRAIN_PIDS}; do
                echo "Killing PID ${PID}..."
                kill ${PID} 2>/dev/null && echo "✅ Killed PID ${PID}" || echo "❌ Failed to kill PID ${PID}"
            done
            sleep 2
            # 강제 종료가 필요한 경우
            REMAINING=$(ps aux | grep "python.*train.py" | grep -v grep | awk '{print $2}')
            if [ -n "${REMAINING}" ]; then
                echo ""
                echo "⚠️  Some processes still running, forcing kill..."
                for PID in ${REMAINING}; do
                    kill -9 ${PID} 2>/dev/null && echo "✅ Force killed PID ${PID}" || echo "❌ Failed to force kill PID ${PID}"
                done
            fi
        else
            echo "Cancelled."
            exit 0
        fi
    fi
else
    # 가장 최근 PID 파일 사용
    LATEST_PID_FILE=$(echo "${PID_FILES}" | head -1)
    PID=$(cat "${LATEST_PID_FILE}" 2>/dev/null)
    
    if [ -z "${PID}" ]; then
        echo "❌ PID file is empty: ${LATEST_PID_FILE}"
        exit 1
    fi
    
    echo "Found PID file: ${LATEST_PID_FILE}"
    echo "PID: ${PID}"
    echo ""
    
    # 프로세스 확인
    if ps -p "${PID}" > /dev/null 2>&1; then
        echo "✅ Training process is running (PID: ${PID})"
        echo ""
        read -p "Kill this process? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Killing PID ${PID}..."
            kill ${PID} 2>/dev/null
            
            # 프로세스가 종료될 때까지 대기
            sleep 2
            
            if ps -p "${PID}" > /dev/null 2>&1; then
                echo "⚠️  Process still running, forcing kill..."
                kill -9 ${PID} 2>/dev/null
                sleep 1
            fi
            
            if ps -p "${PID}" > /dev/null 2>&1; then
                echo "❌ Failed to kill process ${PID}"
                exit 1
            else
                echo "✅ Training process stopped successfully"
                echo ""
                echo "Cleaning up PID file..."
                rm -f "${LATEST_PID_FILE}" && echo "✅ PID file removed" || echo "⚠️  Could not remove PID file"
            fi
        else
            echo "Cancelled."
            exit 0
        fi
    else
        echo "❌ Training process is NOT running (PID ${PID} not found)"
        echo ""
        echo "Cleaning up stale PID file..."
        rm -f "${LATEST_PID_FILE}" && echo "✅ PID file removed" || echo "⚠️  Could not remove PID file"
    fi
fi

echo ""
echo "=========================================="

