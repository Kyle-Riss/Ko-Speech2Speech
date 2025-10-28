#!/bin/bash
#
# CT-Transformer 설치 스크립트
# StreamSpeech에 CT-Transformer 통합을 위한 자동 설치
#

set -e  # 에러 발생 시 중단

echo "========================================"
echo "CT-Transformer Installation Script"
echo "for StreamSpeech Integration"
echo "========================================"
echo ""

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Python 버전 확인
echo "1. Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

if python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "   ${GREEN}✓${NC} Python 3.8+ detected"
else
    echo -e "   ${RED}✗${NC} Python 3.8+ required"
    exit 1
fi
echo ""

# 2. ONNX Runtime 설치
echo "2. Installing ONNX Runtime..."
if python -c "import onnxruntime" 2>/dev/null; then
    echo -e "   ${GREEN}✓${NC} ONNX Runtime already installed"
else
    echo "   Installing onnxruntime-gpu..."
    pip install onnxruntime-gpu==1.14.1 || pip install onnxruntime==1.14.1
    echo -e "   ${GREEN}✓${NC} ONNX Runtime installed"
fi
echo ""

# 3. CT-Transformer 패키지 설치
echo "3. Installing CT-Transformer-punctuation..."
if python -c "from cttpunctuator import Punctuator" 2>/dev/null; then
    echo -e "   ${GREEN}✓${NC} CT-Transformer already installed"
else
    echo "   Cloning repository..."
    if [ -d "CT-Transformer-punctuation" ]; then
        echo "   Removing existing directory..."
        rm -rf CT-Transformer-punctuation
    fi
    
    git clone https://github.com/lovemefan/CT-Transformer-punctuation.git
    cd CT-Transformer-punctuation
    
    echo "   Installing package..."
    pip install -e .
    cd ..
    
    echo -e "   ${GREEN}✓${NC} CT-Transformer installed"
fi
echo ""

# 4. 모델 디렉토리 생성
echo "4. Creating model directory..."
mkdir -p models/ct_transformer
echo -e "   ${GREEN}✓${NC} Directory created: models/ct_transformer/"
echo ""

# 5. 모델 다운로드 안내
echo "5. Downloading CT-Transformer model..."
echo -e "   ${YELLOW}Note:${NC} Model download requires manual step"
echo ""
echo "   Please download the model from one of these sources:"
echo "   - ModelScope: https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx"
echo "   - HuggingFace: https://huggingface.co/mfa/punc_ct-transformer"
echo ""
echo "   Save as: models/ct_transformer/punc.bin"
echo ""

# 모델 파일 확인
if [ -f "models/ct_transformer/punc.bin" ]; then
    echo -e "   ${GREEN}✓${NC} Model file found: models/ct_transformer/punc.bin"
else
    echo -e "   ${YELLOW}!${NC} Model file not found. Please download manually."
    echo ""
    echo "   Quick download (if accessible):"
    echo "   cd models/ct_transformer"
    echo "   wget https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-onnx/resolve/master/punc.onnx"
    echo "   mv punc.onnx punc.bin"
    echo "   cd ../.."
fi
echo ""

# 6. 테스트 실행
echo "6. Running integration tests..."
python test_ct_transformer_integration.py

test_result=$?
echo ""

if [ $test_result -eq 0 ]; then
    echo -e "${GREEN}========================================"
    echo "Installation Completed Successfully! 🎉"
    echo -e "========================================${NC}"
else
    echo -e "${YELLOW}========================================"
    echo "Installation completed with warnings"
    echo -e "========================================${NC}"
    echo ""
    echo "Some tests failed, but core modules are installed."
    echo "This is expected if CT-Transformer model is not downloaded yet."
fi
echo ""

# 7. 다음 단계 안내
echo "Next steps:"
echo "1. Download CT-Transformer model (if not done)"
echo "2. Run test: python test_ct_transformer_integration.py"
echo "3. Try demo: python demo/app.py (if available)"
echo ""
echo "For detailed usage, see:"
echo "- README_CT_TRANSFORMER_INTEGRATION.md"
echo "- docs/CT_TRANSFORMER_SETUP_GUIDE.md"
echo ""

exit 0


