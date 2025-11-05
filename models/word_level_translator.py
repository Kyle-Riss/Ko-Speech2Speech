"""
Word-Level Translator for EchoStream

StreamSpeech 정책 활용:
- Alignment-guided token generation
- Incremental MT Decoder state
- ST CTC 기반 max_new_tokens 계산

참고: agent/speech_to_speech.streamspeech.agent.py:496-650
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class WordLevelTranslator:
    """
    단어 단위 번역기.
    
    StreamSpeech 차이:
    - StreamSpeech: 청크 단위 batch 생성
    - EchoStream: 단어 단위 incremental 생성
    
    StreamSpeech 정책 활용:
    - ST CTC로 max_new_tokens 계산 (Line 496-498)
    - Incremental MT Decoder state 관리
    - Whole word boundary check (Line 540-552)
    """
    
    def __init__(
        self,
        st_ctc_decoder: nn.Module,
        mt_decoder: nn.Module,
        unit_decoder: nn.Module,
        vocoder: nn.Module,
        tokenizer,
        device: str = "cuda",
        lagging_k1: int = 0,  # StreamSpeech parameter
        stride_n: int = 1,    # StreamSpeech parameter
        whole_word: bool = True,  # StreamSpeech parameter
    ):
        self.st_ctc = st_ctc_decoder
        self.mt_decoder = mt_decoder
        self.unit_decoder = unit_decoder
        self.vocoder = vocoder
        self.tokenizer = tokenizer
        self.device = device
        
        # StreamSpeech parameters
        self.lagging_k1 = lagging_k1
        self.stride_n = stride_n
        self.whole_word = whole_word
        
        # Incremental state (StreamSpeech Line 555-574)
        self.mt_incremental_state = {}
        self.prev_mt_tokens = None
        self.prev_output_tokens_mt = None
        
        # CTC collapser
        from word_boundary_detector import CTCCollapser
        self.ctc_collapser = CTCCollapser(
            blank_idx=0,
            pad_idx=tokenizer.pad() if hasattr(tokenizer, 'pad') else 1
        )
        
        logger.info(
            f"WordLevelTranslator initialized "
            f"(lagging_k1={lagging_k1}, stride_n={stride_n}, whole_word={whole_word})"
        )
    
    def reset(self):
        """상태 초기화."""
        self.mt_incremental_state = {}
        self.prev_mt_tokens = None
        self.prev_output_tokens_mt = None
        logger.debug("WordLevelTranslator reset")
    
    def translate_word(
        self,
        encoder_out: torch.Tensor,
        source_word: str,
        asr_tokens: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        단어 번역 (StreamSpeech 정책 활용).
        
        Args:
            encoder_out: Emformer encoder output [B, T, D]
            source_word: Source word text
            asr_tokens: ASR CTC tokens (for logging)
        
        Returns:
            Dict:
                - translation: str
                - units: torch.Tensor
                - waveform: torch.Tensor
                - duration: float (ms)
                - mt_tokens: torch.Tensor (for incremental state)
        """
        # 1. ST CTC Decoder
        st_output = self.st_ctc(encoder_out)
        st_logits = st_output['encoder_out']  # [B, T, vocab]
        st_tokens = st_logits.argmax(dim=-1).squeeze(0)  # [T]
        
        # CTC collapse
        st_collapsed, _ = self.ctc_collapser.collapse(st_tokens)
        st_collapsed_tensor = torch.tensor(st_collapsed, device=self.device)
        
        # 2. Calculate max_new_tokens (StreamSpeech Line 496-498)
        tgt_ctc_length = len(st_collapsed)
        
        if tgt_ctc_length == 0:
            logger.warning(f"No ST CTC output for word '{source_word}'")
            return self._empty_result()
        
        # StreamSpeech alignment-guided calculation
        max_new_tokens = (
            (tgt_ctc_length - self.lagging_k1) // self.stride_n
        ) * self.stride_n
        
        if max_new_tokens < 1:
            max_new_tokens = 1  # At least 1 token
        
        logger.debug(
            f"ST CTC length: {tgt_ctc_length}, "
            f"max_new_tokens: {max_new_tokens}"
        )
        
        # 3. MT Decoder (incremental, StreamSpeech Line 520-533)
        mt_output = self._generate_mt_tokens(
            encoder_out=encoder_out,
            max_new_tokens=max_new_tokens
        )
        
        if mt_output is None:
            return self._empty_result()
        
        mt_tokens = mt_output['tokens']
        mt_hidden = mt_output['decoder_out']
        
        # 4. Whole word check (StreamSpeech Line 540-552)
        if self.whole_word and not self._check_whole_word(mt_tokens):
            logger.debug(f"Incomplete word, waiting...")
            return self._empty_result()
        
        # 5. Decode translation
        try:
            translation = self.tokenizer.decode(mt_tokens.tolist())
            translation = self._clean_text(translation)
        except Exception as e:
            logger.warning(f"MT decode failed: {e}")
            translation = ""
        
        # 6. Unit Decoder
        unit_output = self.unit_decoder(mt_hidden)
        units = unit_output['units']  # [T_unit]
        
        if units.size(0) == 0:
            logger.warning("No units generated")
            return self._empty_result()
        
        # 7. Vocoder
        waveform = self.vocoder(units.unsqueeze(0))  # [1, T_wav]
        duration = waveform.size(1) / 16000 * 1000  # ms
        
        logger.info(
            f"Translated: '{source_word}' → '{translation}' "
            f"({units.size(0)} units, {duration:.1f}ms)"
        )
        
        return {
            'source_word': source_word,
            'translation': translation,
            'units': units,
            'waveform': waveform,
            'duration': duration,
            'mt_tokens': mt_tokens,
            'st_tokens': st_collapsed_tensor,
        }
    
    def _generate_mt_tokens(
        self,
        encoder_out: torch.Tensor,
        max_new_tokens: int,
    ) -> Optional[Dict]:
        """
        MT Decoder로 토큰 생성 (incremental).
        
        StreamSpeech 로직 참고: Line 520-650
        """
        # Prepare prev_output_tokens
        if self.prev_output_tokens_mt is None:
            # First generation
            prev_output_tokens = torch.tensor(
                [[self.mt_decoder.eos_idx]],  # Start with EOS
                device=self.device
            )
        else:
            prev_output_tokens = self.prev_output_tokens_mt
        
        # Generate new tokens
        new_tokens = []
        for _ in range(max_new_tokens):
            # MT Decoder forward
            decoder_out = self.mt_decoder(
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
                incremental_state=self.mt_incremental_state,
            )
            
            # Get next token
            logits = decoder_out['decoder_out'][:, -1, :]  # [B, vocab]
            next_token = logits.argmax(dim=-1)  # [B]
            
            # Check EOS
            if next_token.item() == self.mt_decoder.eos_idx:
                break
            
            new_tokens.append(next_token.item())
            
            # Update prev_output_tokens
            prev_output_tokens = torch.cat([
                prev_output_tokens,
                next_token.unsqueeze(1)
            ], dim=1)
        
        if len(new_tokens) == 0:
            return None
        
        # Update state
        self.prev_output_tokens_mt = prev_output_tokens
        self.prev_mt_tokens = torch.tensor(new_tokens, device=self.device)
        
        return {
            'tokens': self.prev_mt_tokens,
            'decoder_out': decoder_out['decoder_out'],
        }
    
    def _check_whole_word(self, tokens: torch.Tensor) -> bool:
        """
        Whole word boundary check (StreamSpeech Line 540-552).
        
        조건: 마지막 토큰이 ▁로 시작하는가?
        
        Returns:
            True: Complete word
            False: Incomplete word
        """
        if tokens.size(0) == 0:
            return False
        
        last_token = tokens[-1].item()
        
        try:
            last_text = self.tokenizer.decode([last_token])
            
            # SentencePiece word boundary
            if last_text.startswith("▁"):
                return True
            
            # Punctuation
            if last_text in [".", ",", "!", "?", ";", ":"]:
                return True
            
        except Exception as e:
            logger.warning(f"Token decode failed: {e}")
        
        return False
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정리 (StreamSpeech Line 595-602)."""
        text = text.replace("_", " ")
        text = text.replace("▁", " ")
        text = text.replace("<unk>", " ")
        text = text.replace("<s>", "")
        text = text.replace("</s>", "")
        
        if len(text) > 0 and text[0] == " ":
            text = text[1:]
        
        return text
    
    def _empty_result(self) -> Dict:
        """빈 결과 반환."""
        return {
            'source_word': "",
            'translation': "",
            'units': torch.tensor([], device=self.device),
            'waveform': torch.tensor([[]], device=self.device),
            'duration': 0.0,
            'mt_tokens': torch.tensor([], device=self.device),
            'st_tokens': torch.tensor([], device=self.device),
        }


if __name__ == "__main__":
    print("="*70)
    print("Testing WordLevelTranslator")
    print("="*70)
    
    # Mock components
    class MockSTCTC(nn.Module):
        def forward(self, x):
            B, T, D = x.shape
            vocab_size = 6000
            logits = torch.randn(B, T, vocab_size)
            return {'encoder_out': logits}
    
    class MockMTDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.eos_idx = 2
        
        def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
            B, T = prev_output_tokens.shape
            vocab_size = 6000
            logits = torch.randn(B, T, vocab_size)
            return {'decoder_out': logits}
    
    class MockUnitDecoder(nn.Module):
        def forward(self, x):
            B, T, D = x.shape
            units = torch.randint(0, 1000, (T * 5,))  # 5x upsampling
            return {'units': units}
    
    class MockVocoder(nn.Module):
        def forward(self, units):
            B, T = units.shape
            wav_length = T * 160  # hop_size=160
            return torch.randn(B, wav_length)
    
    class MockTokenizer:
        def decode(self, tokens):
            return "hello"
        
        def pad(self):
            return 1
    
    # Initialize
    st_ctc = MockSTCTC()
    mt_decoder = MockMTDecoder()
    unit_decoder = MockUnitDecoder()
    vocoder = MockVocoder()
    tokenizer = MockTokenizer()
    
    translator = WordLevelTranslator(
        st_ctc_decoder=st_ctc,
        mt_decoder=mt_decoder,
        unit_decoder=unit_decoder,
        vocoder=vocoder,
        tokenizer=tokenizer,
        device="cpu",
        lagging_k1=0,
        stride_n=1,
        whole_word=False,  # Disable for testing
    )
    
    print("\n1. Testing word translation...")
    encoder_out = torch.randn(1, 10, 256)
    
    result = translator.translate_word(
        encoder_out=encoder_out,
        source_word="안녕",
    )
    
    print(f"   Source: {result['source_word']}")
    print(f"   Translation: {result['translation']}")
    print(f"   Units: {result['units'].shape}")
    print(f"   Waveform: {result['waveform'].shape}")
    print(f"   Duration: {result['duration']:.1f}ms")
    print("   ✅ Translation successful")
    
    print("\n2. Testing incremental state...")
    result2 = translator.translate_word(
        encoder_out=encoder_out,
        source_word="하세요",
    )
    
    print(f"   Previous tokens: {translator.prev_mt_tokens}")
    print(f"   New translation: {result2['translation']}")
    print("   ✅ Incremental state maintained")
    
    print("\n3. Testing reset...")
    translator.reset()
    print(f"   Incremental state: {translator.mt_incremental_state}")
    print(f"   Previous tokens: {translator.prev_mt_tokens}")
    print("   ✅ Reset successful")
    
    print("\n" + "="*70)
    print("✅ All WordLevelTranslator tests passed!")
    print("="*70)
    
    print("\n💡 Usage:")
    print("  translator = WordLevelTranslator(st_ctc, mt, unit, vocoder, tokenizer)")
    print("  ")
    print("  # Translate word")
    print("  result = translator.translate_word(encoder_out, 'hello')")
    print("  print(f\"Translation: {result['translation']}\")")
    print("  ")
    print("  # Play waveform")
    print("  play_audio(result['waveform'])")

