#!/bin/bash
# EchoStream 데모 서버 시작 스크립트
# 사용법: ./start_demo.sh [hello|yes|no]

PHRASE=${1:-hello}  # 기본값: hello

case $PHRASE in
    hello)
        UNITS_FILE="/Users/hayubin/EchoStream/forced_units/hello.npy"
        ;;
    yes)
        UNITS_FILE="/Users/hayubin/EchoStream/forced_units/yes.npy"
        ;;
    no)
        UNITS_FILE="/Users/hayubin/EchoStream/forced_units/no.npy"
        ;;
    *)
        echo "❌ 알 수 없는 문구: $PHRASE (사용 가능: hello, yes, no)"
        exit 1
        ;;
esac

if [ ! -f "$UNITS_FILE" ]; then
    echo "❌ 파일이 없습니다: $UNITS_FILE"
    echo "   먼저 collect_easy_units.py를 실행하세요."
    exit 1
fi

echo "🚀 EchoStream 데모 서버 시작..."
echo "   강제 유닛: $PHRASE ($UNITS_FILE)"
echo ""

export ECHOSTREAM_FORCE_VOCODER=1
export ECHOSTREAM_FORCED_UNITS="$UNITS_FILE"

uvicorn server.fastapi_app:app --host 0.0.0.0 --port 8000


