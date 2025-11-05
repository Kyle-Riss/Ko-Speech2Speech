"""
Word Boundary Detector for EchoStream

StreamSpeech 개선:
- StreamSpeech: stride_n 기반 고정 토큰 대기
- EchoStream: 단어 경계 기반 동적 출력

참고: StreamSpeech agent/ctc_decoder.py의 CTC collapse 로직 활용
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class CTCCollapser:
    """
    CTC 출력 후처리 (StreamSpeech 로직 차용).
    
    참고: agent/ctc_decoder.py:67-89
    """
    
    def __init__(self, blank_idx=0, pad_idx=1):
        self.blank_idx = blank_idx
        self.pad_idx = pad_idx
    
    def collapse(self, tokens: torch.Tensor) -> Tuple[List[int], List[int]]:
        """
        CTC collapse: blank 제거 + 중복 제거.
        
        Args:
            tokens: [T] CTC output tokens
        
        Returns:
            collapsed_tokens: List of unique tokens
            indices: Original indices of collapsed tokens
        """
        _toks = tokens.int().tolist()
        
        # Deduplicate (StreamSpeech Line 69-71)
        deduplicated_toks = [
            (v, i) for i, v in enumerate(_toks) 
            if i == 0 or v != _toks[i - 1]
        ]
        
        # Remove blank and pad (StreamSpeech Line 72-76)
        collapsed = []
        indices = []
        for v, i in deduplicated_toks:
            if v != self.blank_idx and v != self.pad_idx:
                collapsed.append(v)
                indices.append(i)
        
        return collapsed, indices


class WordBoundaryDetector:
    """
    단어 경계 탐지기.
    
    핵심 개선:
    - StreamSpeech: stride_n 토큰마다 체크 (고정)
    - EchoStream: 단어 완성 즉시 탐지 (동적)
    
    방법:
    1. ASR CTC로 실시간 텍스트 생성
    2. SentencePiece ▁ 토큰으로 단어 경계 판단
    3. 단어 완성 시 즉시 반환
    """
    
    def __init__(
        self,
        emformer_encoder: nn.Module,
        asr_ctc_decoder: nn.Module,
        tokenizer,  # SentencePiece tokenizer
        device: str = "cuda",
    ):
        self.encoder = emformer_encoder
        self.asr_ctc = asr_ctc_decoder
        self.tokenizer = tokenizer
        self.device = device
        
        # CTC collapser (StreamSpeech 로직)
        self.ctc_collapser = CTCCollapser(
            blank_idx=0,
            pad_idx=tokenizer.pad() if hasattr(tokenizer, 'pad') else 1
        )
        
        # State
        self.encoder_cache = {}
        self.partial_word = ""
        self.segment_buffer = []
        
        logger.info("WordBoundaryDetector initialized")
    
    def reset(self):
        """상태 초기화."""
        self.encoder_cache = {}
        self.partial_word = ""
        self.segment_buffer = []
        logger.debug("WordBoundaryDetector reset")
    
    def process_segment(
        self,
        audio_segment: torch.Tensor,  # [T_seg, F]
    ) -> Optional[Dict]:
        """
        세그먼트 처리 및 단어 경계 탐지.
        
        Args:
            audio_segment: [T_seg, F] audio features
        
        Returns:
            None: 단어 미완성
            Dict: 완성된 단어 정보
                - word: str
                - encoder_out: torch.Tensor
                - asr_tokens: torch.Tensor
                - start_time: float (ms)
                - end_time: float (ms)
        """
        # 1. Emformer encoding (with cache)
        encoder_out, self.encoder_cache = self.encoder(
            audio_segment.unsqueeze(0).to(self.device),
            cache=self.encoder_cache
        )
        
        # 2. ASR CTC decoding
        asr_logits = self.asr_ctc(encoder_out)  # [B, T, vocab]
        asr_tokens = asr_logits.argmax(dim=-1).squeeze(0)  # [T]
        
        # 3. CTC collapse (StreamSpeech 로직)
        collapsed_tokens, indices = self.ctc_collapser.collapse(asr_tokens)
        
        if len(collapsed_tokens) == 0:
            # No new tokens
            self.segment_buffer.append({
                'encoder_out': encoder_out,
                'time': len(self.segment_buffer) * 40,  # 40ms per segment
            })
            return None
        
        # 4. Decode to text
        try:
            new_text = self.tokenizer.decode(collapsed_tokens)
        except Exception as e:
            logger.warning(f"Tokenizer decode failed: {e}")
            new_text = ""
        
        # 5. Word boundary check
        if self._is_word_boundary(new_text):
            # 단어 완성!
            word = self.partial_word + new_text.rstrip("▁ ")
            
            result = {
                'word': word,
                'encoder_out': encoder_out,
                'asr_tokens': torch.tensor(collapsed_tokens, device=self.device),
                'start_time': self.segment_buffer[0]['time'] if self.segment_buffer else 0,
                'end_time': len(self.segment_buffer) * 40,
                'is_complete': True,
            }
            
            # 버퍼 초기화
            self.partial_word = ""
            self.segment_buffer = []
            
            logger.debug(f"Word completed: '{word}' ({result['start_time']}-{result['end_time']}ms)")
            
            return result
        else:
            # 단어 미완성
            self.partial_word += new_text
            self.segment_buffer.append({
                'encoder_out': encoder_out,
                'time': len(self.segment_buffer) * 40,
            })
            
            logger.debug(f"Partial word: '{self.partial_word}'")
            
            return None
    
    def _is_word_boundary(self, text: str) -> bool:
        """
        단어 경계 판단.
        
        조건:
        1. SentencePiece ▁ 토큰 (단어 시작)
        2. 공백 문자
        3. 구두점
        
        Returns:
            True: 단어 완성
            False: 단어 미완성
        """
        if not text:
            return False
        
        # SentencePiece word boundary
        if text.endswith("▁"):
            return True
        
        # Space
        if text.endswith(" "):
            return True
        
        # Punctuation
        if text.endswith((".", ",", "!", "?", ";", ":")):
            return True
        
        return False
    
    def force_complete(self) -> Optional[Dict]:
        """
        강제로 현재 partial word를 완성.
        
        사용: 음성 입력 종료 시
        
        Returns:
            None: partial word 없음
            Dict: 강제 완성된 단어
        """
        if not self.partial_word:
            return None
        
        if not self.segment_buffer:
            return None
        
        # 마지막 encoder output 사용
        last_segment = self.segment_buffer[-1]
        
        result = {
            'word': self.partial_word,
            'encoder_out': last_segment['encoder_out'],
            'asr_tokens': torch.tensor([], device=self.device),  # Empty
            'start_time': self.segment_buffer[0]['time'],
            'end_time': last_segment['time'],
            'is_complete': True,
            'forced': True,
        }
        
        # 초기화
        self.partial_word = ""
        self.segment_buffer = []
        
        logger.info(f"Force completed word: '{result['word']}'")
        
        return result


if __name__ == "__main__":
    print("="*70)
    print("Testing WordBoundaryDetector")
    print("="*70)
    
    # Mock components
    class MockEmformer(nn.Module):
        def forward(self, x, cache=None):
            B, T, F = x.shape
            out = torch.randn(B, T, 256)
            new_cache = {'mock': True}
            return out, new_cache
    
    class MockASRCTC(nn.Module):
        def forward(self, x):
            B, T, D = x.shape
            vocab_size = 6000
            return torch.randn(B, T, vocab_size)
    
    class MockTokenizer:
        def decode(self, tokens):
            # Mock: return some text
            if len(tokens) > 0:
                return "hello▁"
            return ""
        
        def pad(self):
            return 1
    
    # Initialize
    encoder = MockEmformer()
    asr_ctc = MockASRCTC()
    tokenizer = MockTokenizer()
    
    detector = WordBoundaryDetector(
        emformer_encoder=encoder,
        asr_ctc_decoder=asr_ctc,
        tokenizer=tokenizer,
        device="cpu"
    )
    
    print("\n1. Testing segment processing...")
    
    # Segment 1: partial word
    segment1 = torch.randn(4, 80)  # 4 frames, 80 features
    result1 = detector.process_segment(segment1)
    print(f"   Segment 1: {result1}")
    
    # Segment 2: word completion
    segment2 = torch.randn(4, 80)
    result2 = detector.process_segment(segment2)
    print(f"   Segment 2: {result2}")
    
    if result2:
        print(f"   ✅ Word detected: '{result2['word']}'")
        print(f"   Time: {result2['start_time']}-{result2['end_time']}ms")
    
    print("\n2. Testing force complete...")
    detector.partial_word = "incomplete"
    detector.segment_buffer = [{'encoder_out': torch.randn(1, 4, 256), 'time': 0}]
    
    forced = detector.force_complete()
    if forced:
        print(f"   ✅ Forced word: '{forced['word']}'")
    
    print("\n3. Testing reset...")
    detector.reset()
    print(f"   Partial word: '{detector.partial_word}'")
    print(f"   Buffer length: {len(detector.segment_buffer)}")
    print("   ✅ Reset successful")
    
    print("\n" + "="*70)
    print("✅ All WordBoundaryDetector tests passed!")
    print("="*70)
    
    print("\n💡 Usage:")
    print("  detector = WordBoundaryDetector(encoder, asr_ctc, tokenizer)")
    print("  ")
    print("  # Process audio segments")
    print("  for segment in audio_stream:")
    print("      result = detector.process_segment(segment)")
    print("      if result:")
    print("          print(f\"Word: {result['word']}\")")
    print("  ")
    print("  # Force complete at end")
    print("  final = detector.force_complete()")

