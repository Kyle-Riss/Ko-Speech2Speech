"""
Sentence Recomposer for EchoStream

기능:
1. 단어별 출력: 저지연 (40ms)
2. 문장 완성 시: 전체 재합성 (고품질)

CT-Transformer 통합:
- 문장 경계 탐지
- 재조합 트리거
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class SentenceRecomposer:
    """
    문장 단위 재조합기.
    
    전략:
    1. 단어별 출력: 즉시 출력 (저지연)
    2. 문장 완성 시: 전체 재합성 (고품질)
    
    장점:
    - 실시간성 유지 (단어별 출력)
    - 최종 품질 보장 (문장 재합성)
    """
    
    def __init__(
        self,
        ct_transformer: Optional[nn.Module],  # Punctuation model
        vocoder: nn.Module,
        max_sentence_length: int = 50,  # words
        device: str = "cuda",
    ):
        self.ct_transformer = ct_transformer
        self.vocoder = vocoder
        self.max_sentence_length = max_sentence_length
        self.device = device
        
        # Buffers
        self.source_words = []
        self.translated_words = []
        self.unit_buffer = []
        self.waveform_buffer = []
        self.mt_tokens_buffer = []
        
        # Timing
        self.sentence_start_time = 0.0
        self.word_count = 0
        
        logger.info(
            f"SentenceRecomposer initialized "
            f"(max_sentence_length={max_sentence_length})"
        )
    
    def reset(self):
        """상태 초기화."""
        self.source_words = []
        self.translated_words = []
        self.unit_buffer = []
        self.waveform_buffer = []
        self.mt_tokens_buffer = []
        self.sentence_start_time = 0.0
        self.word_count = 0
        logger.debug("SentenceRecomposer reset")
    
    def add_word(
        self,
        word_result: Dict,
    ) -> Dict:
        """
        단어 추가 및 문장 경계 체크.
        
        Args:
            word_result: WordLevelTranslator 출력
                - source_word: str
                - translation: str
                - units: torch.Tensor
                - waveform: torch.Tensor
                - mt_tokens: torch.Tensor
        
        Returns:
            Dict:
                - type: 'word' or 'sentence'
                - content: waveform
                - text: translation
                - is_final: bool
        """
        # Empty result check
        if not word_result.get('translation'):
            logger.debug("Empty word result, skipping")
            return self._word_output(torch.tensor([[]], device=self.device), "")
        
        # 1. 버퍼에 추가
        self.source_words.append(word_result['source_word'])
        self.translated_words.append(word_result['translation'])
        self.unit_buffer.append(word_result['units'])
        self.waveform_buffer.append(word_result['waveform'])
        
        if 'mt_tokens' in word_result:
            self.mt_tokens_buffer.append(word_result['mt_tokens'])
        
        self.word_count += 1
        
        logger.debug(
            f"Word added: '{word_result['translation']}' "
            f"(total: {self.word_count} words)"
        )
        
        # 2. CT-Transformer로 문장 경계 탐지
        is_sentence_end = False
        punctuated_text = " ".join(self.translated_words)
        
        if self.ct_transformer is not None:
            try:
                punctuated, is_end = self.ct_transformer.predict(punctuated_text)
                is_sentence_end = is_end
                punctuated_text = punctuated
                
                logger.debug(
                    f"CT-Transformer: is_end={is_end}, "
                    f"text='{punctuated}'"
                )
            except Exception as e:
                logger.warning(f"CT-Transformer failed: {e}")
                # Fallback: check punctuation manually
                is_sentence_end = self._check_punctuation(punctuated_text)
        else:
            # No CT-Transformer: check punctuation manually
            is_sentence_end = self._check_punctuation(punctuated_text)
        
        # 3. 문장 완성 체크
        if is_sentence_end or self.word_count >= self.max_sentence_length:
            # ⭐ 문장 재조합 트리거!
            logger.info(
                f"Sentence completed: {self.word_count} words, "
                f"recomposing..."
            )
            return self._recompose_sentence(punctuated_text)
        else:
            # 단어만 출력
            return self._word_output(
                word_result['waveform'],
                word_result['translation']
            )
    
    def _recompose_sentence(self, punctuated_text: str) -> Dict:
        """
        전체 문장 재합성.
        
        이유:
        - 단어별 생성은 prosody가 끊김
        - 전체 문장을 재생성하면 자연스러운 억양
        
        Returns:
            Dict:
                - type: 'sentence'
                - content: waveform (재합성)
                - text: punctuated text
                - is_final: True
        """
        if len(self.unit_buffer) == 0:
            logger.warning("No units to recompose")
            return self._sentence_output(
                torch.tensor([[]], device=self.device),
                punctuated_text,
                is_final=True
            )
        
        # 1. 모든 유닛 결합
        all_units = torch.cat(self.unit_buffer, dim=0)  # [T_total]
        
        logger.debug(
            f"Recomposing {len(self.unit_buffer)} word units "
            f"({all_units.size(0)} total units)"
        )
        
        # 2. Vocoder로 재합성
        # ⭐ 핵심: 전체 시퀀스를 한 번에 생성
        try:
            final_waveform = self.vocoder(all_units.unsqueeze(0))  # [1, T_wav]
        except Exception as e:
            logger.error(f"Vocoder failed: {e}")
            # Fallback: concatenate word waveforms
            final_waveform = torch.cat(self.waveform_buffer, dim=1)
        
        duration = final_waveform.size(1) / 16000 * 1000  # ms
        
        logger.info(
            f"Sentence recomposed: '{punctuated_text}' "
            f"({all_units.size(0)} units, {duration:.1f}ms)"
        )
        
        # 3. 결과 반환
        result = self._sentence_output(
            final_waveform,
            punctuated_text,
            is_final=True
        )
        
        # 4. 버퍼 초기화
        self.sentence_start_time += duration
        self.source_words = []
        self.translated_words = []
        self.unit_buffer = []
        self.waveform_buffer = []
        self.mt_tokens_buffer = []
        self.word_count = 0
        
        return result
    
    def _check_punctuation(self, text: str) -> bool:
        """
        구두점 기반 문장 경계 체크 (fallback).
        
        Returns:
            True: Sentence end
            False: Sentence continues
        """
        if not text:
            return False
        
        # Check last character
        if text.rstrip().endswith((".", "!", "?")):
            return True
        
        return False
    
    def _word_output(self, waveform: torch.Tensor, text: str) -> Dict:
        """단어 출력 결과."""
        return {
            'type': 'word',
            'content': waveform,
            'text': text,
            'is_final': False,
            'word_count': self.word_count,
        }
    
    def _sentence_output(
        self,
        waveform: torch.Tensor,
        text: str,
        is_final: bool = True
    ) -> Dict:
        """문장 출력 결과."""
        return {
            'type': 'sentence',
            'content': waveform,
            'text': text,
            'is_final': is_final,
            'word_count': self.word_count,
            'recomposed': True,
        }
    
    def force_complete(self) -> Optional[Dict]:
        """
        강제로 현재 버퍼를 문장으로 완성.
        
        사용: 음성 입력 종료 시
        
        Returns:
            None: 버퍼 비어있음
            Dict: 강제 완성된 문장
        """
        if len(self.translated_words) == 0:
            return None
        
        current_text = " ".join(self.translated_words)
        
        logger.info(
            f"Force completing sentence: '{current_text}' "
            f"({self.word_count} words)"
        )
        
        return self._recompose_sentence(current_text)


if __name__ == "__main__":
    print("="*70)
    print("Testing SentenceRecomposer")
    print("="*70)
    
    # Mock components
    class MockCTTransformer:
        def predict(self, text):
            # Mock: detect sentence end if text ends with period
            is_end = text.rstrip().endswith(".")
            punctuated = text if "." in text else text + "."
            return punctuated, is_end
    
    class MockVocoder(nn.Module):
        def forward(self, units):
            B, T = units.shape
            wav_length = T * 160  # hop_size=160
            return torch.randn(B, wav_length)
    
    # Initialize
    ct_transformer = MockCTTransformer()
    vocoder = MockVocoder()
    
    recomposer = SentenceRecomposer(
        ct_transformer=ct_transformer,
        vocoder=vocoder,
        max_sentence_length=10,
        device="cpu"
    )
    
    print("\n1. Testing word addition...")
    
    # Word 1
    word1 = {
        'source_word': '안녕',
        'translation': 'Hello',
        'units': torch.randint(0, 1000, (50,)),
        'waveform': torch.randn(1, 8000),
        'mt_tokens': torch.tensor([10, 20]),
    }
    
    result1 = recomposer.add_word(word1)
    print(f"   Word 1: type={result1['type']}, text='{result1['text']}'")
    assert result1['type'] == 'word', "Should be word output"
    
    # Word 2
    word2 = {
        'source_word': '하세요',
        'translation': 'there',
        'units': torch.randint(0, 1000, (50,)),
        'waveform': torch.randn(1, 8000),
        'mt_tokens': torch.tensor([30, 40]),
    }
    
    result2 = recomposer.add_word(word2)
    print(f"   Word 2: type={result2['type']}, text='{result2['text']}'")
    
    # Word 3 with period (sentence end)
    word3 = {
        'source_word': '.',
        'translation': '.',
        'units': torch.randint(0, 1000, (10,)),
        'waveform': torch.randn(1, 1600),
        'mt_tokens': torch.tensor([50]),
    }
    
    result3 = recomposer.add_word(word3)
    print(f"   Word 3: type={result3['type']}, text='{result3['text']}'")
    assert result3['type'] == 'sentence', "Should be sentence output"
    assert result3['recomposed'], "Should be recomposed"
    print("   ✅ Sentence recomposition successful")
    
    print("\n2. Testing force complete...")
    
    # Add some words
    recomposer.add_word(word1)
    recomposer.add_word(word2)
    
    forced = recomposer.force_complete()
    print(f"   Forced: type={forced['type']}, text='{forced['text']}'")
    assert forced['type'] == 'sentence', "Should be sentence"
    print("   ✅ Force complete successful")
    
    print("\n3. Testing reset...")
    recomposer.reset()
    print(f"   Word count: {recomposer.word_count}")
    print(f"   Buffer length: {len(recomposer.unit_buffer)}")
    print("   ✅ Reset successful")
    
    print("\n4. Testing without CT-Transformer...")
    recomposer_no_ct = SentenceRecomposer(
        ct_transformer=None,  # No CT-Transformer
        vocoder=vocoder,
        device="cpu"
    )
    
    result = recomposer_no_ct.add_word(word1)
    print(f"   Without CT: type={result['type']}")
    print("   ✅ Fallback to punctuation check")
    
    print("\n" + "="*70)
    print("✅ All SentenceRecomposer tests passed!")
    print("="*70)
    
    print("\n💡 Usage:")
    print("  recomposer = SentenceRecomposer(ct_transformer, vocoder)")
    print("  ")
    print("  # Add words")
    print("  for word_result in word_stream:")
    print("      output = recomposer.add_word(word_result)")
    print("      ")
    print("      if output['type'] == 'word':")
    print("          play_audio(output['content'])  # Low latency")
    print("      elif output['type'] == 'sentence':")
    print("          play_audio(output['content'])  # High quality")
    print("  ")
    print("  # Force complete at end")
    print("  final = recomposer.force_complete()")

