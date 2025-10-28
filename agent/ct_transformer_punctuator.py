"""
CT-Transformer Punctuation Predictor for StreamSpeech
Based on: https://github.com/lovemefan/CT-Transformer-punctuation

Integrates CT-Transformer to predict punctuation in real-time ASR output
and provides sentence boundary signals for re-composition triggering.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class CTTransformerPunctuator:
    """
    CT-Transformer based punctuation predictor for real-time streaming.
    
    Predicts punctuation marks and detects sentence boundaries to trigger
    re-composition in StreamSpeech pipeline.
    """
    
    def __init__(self, model_path: str, mode: str = "online"):
        """
        Initialize CT-Transformer punctuator.
        
        Args:
            model_path: Path to ONNX model file (punc.bin)
            mode: "online" for streaming or "offline" for batch processing
        """
        self.mode = mode
        self.model_path = model_path
        
        try:
            from cttpunctuator.punctuator import Punctuator as CTTPunctuator
            logger.info(f"Initializing CT-Transformer punctuator with {mode} mode.")
            self.punctuator = CTTPunctuator(model_path, mode=mode)
            logger.info(f"{mode.capitalize()} model initialized.")
        except ImportError:
            logger.error(
                "CT-Transformer-punctuation not found. "
                "Please install: pip install git+https://github.com/lovemefan/CT-Transformer-punctuation.git"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize CT-Transformer: {e}")
            raise
        
        # 문장 종결 구두점
        self.sentence_terminators = {'.', '?', '!', '。', '？', '！'}
        
        # 온라인 모드용 캐시
        self.cache = {} if mode == "online" else None
        
        logger.info("CT-Transformer punctuator initialized successfully.")
    
    def predict(
        self, 
        text: str, 
        param_dict: Optional[Dict] = None
    ) -> Tuple[str, bool, List[str]]:
        """
        Predict punctuation for input text.
        
        Args:
            text: Input text without punctuation
            param_dict: Parameters for online mode (cache, etc.)
        
        Returns:
            Tuple of:
                - punctuated_text: Text with predicted punctuation
                - is_sentence_end: True if sentence terminator detected
                - sentence_terminators_found: List of terminator positions
        """
        if not text or len(text.strip()) == 0:
            return "", False, []
        
        # 온라인 모드용 파라미터
        if param_dict is None:
            param_dict = {"cache": []} if self.mode == "online" else {}
        
        try:
            # CT-Transformer로 구두점 예측
            if self.mode == "online":
                result = self.punctuator.punctuate(text, param_dict=param_dict)
                punctuated_text = result[0] if isinstance(result, (list, tuple)) else result
            else:
                punctuated_text = self.punctuator.punctuate(text)
            
            # 문장 종결 구두점 탐지
            is_sentence_end, terminators = self._detect_sentence_boundaries(
                punctuated_text
            )
            
            return punctuated_text, is_sentence_end, terminators
            
        except Exception as e:
            logger.error(f"CT-Transformer prediction failed: {e}")
            return text, False, []
    
    def _detect_sentence_boundaries(
        self, 
        punctuated_text: str
    ) -> Tuple[bool, List[str]]:
        """
        Detect sentence boundaries in punctuated text.
        
        Args:
            punctuated_text: Text with predicted punctuation
        
        Returns:
            Tuple of:
                - is_sentence_end: True if ends with terminator
                - terminators_found: List of all terminators found
        """
        terminators_found = []
        
        # 마지막 문자 확인
        stripped = punctuated_text.strip()
        is_sentence_end = False
        
        if stripped:
            last_char = stripped[-1]
            if last_char in self.sentence_terminators:
                is_sentence_end = True
                terminators_found.append(last_char)
        
        # 모든 종결 구두점 위치 찾기
        for i, char in enumerate(punctuated_text):
            if char in self.sentence_terminators:
                if char not in terminators_found:
                    terminators_found.append(char)
        
        return is_sentence_end, terminators_found
    
    def reset_cache(self):
        """Reset online mode cache."""
        if self.cache is not None:
            self.cache = {}


class SentenceBoundaryDetector:
    """
    Sentence boundary detector using CT-Transformer for StreamSpeech.
    
    Buffers ASR text and triggers re-composition when sentence boundaries
    are detected.
    """
    
    def __init__(
        self, 
        punctuator: CTTransformerPunctuator,
        buffer_size: int = 50,
        min_trigger_length: int = 5
    ):
        """
        Initialize sentence boundary detector.
        
        Args:
            punctuator: CT-Transformer punctuator instance
            buffer_size: Maximum buffer size for ASR text
            min_trigger_length: Minimum text length to trigger prediction
        """
        self.punctuator = punctuator
        self.buffer_size = buffer_size
        self.min_trigger_length = min_trigger_length
        
        # ASR 텍스트 버퍼
        self.text_buffer = ""
        
        # 온라인 모드용 파라미터
        self.param_dict = {"cache": []}
        
        # 통계
        self.total_predictions = 0
        self.sentences_detected = 0
        
        logger.info(
            f"Sentence boundary detector initialized "
            f"(buffer_size={buffer_size}, min_trigger_length={min_trigger_length})"
        )
    
    def add_text(self, new_text: str) -> Tuple[bool, str, str]:
        """
        Add new ASR text and check for sentence boundaries.
        
        Args:
            new_text: New ASR text to add
        
        Returns:
            Tuple of:
                - trigger: True if sentence boundary detected
                - complete_sentence: Complete sentence if triggered
                - remaining_text: Remaining text in buffer
        """
        # 버퍼에 추가
        self.text_buffer += new_text
        
        # 버퍼 크기 제한
        if len(self.text_buffer) > self.buffer_size:
            self.text_buffer = self.text_buffer[-self.buffer_size:]
        
        # 최소 길이 확인
        if len(self.text_buffer) < self.min_trigger_length:
            return False, "", self.text_buffer
        
        # CT-Transformer로 구두점 예측
        punctuated, is_end, terminators = self.punctuator.predict(
            self.text_buffer,
            param_dict=self.param_dict
        )
        
        self.total_predictions += 1
        
        # 문장 종결 감지
        if is_end:
            self.sentences_detected += 1
            
            # 완성된 문장과 남은 텍스트 분리
            complete_sentence = punctuated
            self.text_buffer = ""  # 버퍼 초기화
            
            logger.info(
                f"Sentence boundary detected: '{complete_sentence}' "
                f"(terminators: {terminators})"
            )
            
            return True, complete_sentence, ""
        
        return False, "", self.text_buffer
    
    def force_complete(self) -> str:
        """
        Force complete current buffer (for end of stream).
        
        Returns:
            Complete sentence with predicted punctuation
        """
        if not self.text_buffer:
            return ""
        
        punctuated, _, _ = self.punctuator.predict(
            self.text_buffer,
            param_dict=self.param_dict
        )
        
        self.text_buffer = ""
        return punctuated
    
    def reset(self):
        """Reset detector state."""
        self.text_buffer = ""
        self.param_dict = {"cache": []}
        self.punctuator.reset_cache()
    
    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return {
            "total_predictions": self.total_predictions,
            "sentences_detected": self.sentences_detected,
            "buffer_length": len(self.text_buffer),
            "detection_rate": (
                self.sentences_detected / max(1, self.total_predictions)
            )
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize punctuator
    model_path = "path/to/cttpunctuator/src/onnx/punc.bin"
    punctuator = CTTransformerPunctuator(model_path, mode="online")
    
    # Initialize boundary detector
    detector = SentenceBoundaryDetector(punctuator)
    
    # Simulate streaming ASR text
    asr_chunks = [
        "跨境河流是养育沿岸",
        "人民的生命之源",
        "长期以来为帮助下游地区防灾减灾",
        "中方技术人员在上游地区",
        "极为恶劣的自然条件下",
        "克服巨大困难",
    ]
    
    for chunk in asr_chunks:
        trigger, sentence, remaining = detector.add_text(chunk)
        
        if trigger:
            print(f"✓ Complete sentence: {sentence}")
            print(f"  Remaining: {remaining}")
        else:
            print(f"  Buffering: {remaining}")
    
    # Force complete at end
    final = detector.force_complete()
    if final:
        print(f"✓ Final sentence: {final}")
    
    # Print stats
    print(f"\nStats: {detector.get_stats()}")


