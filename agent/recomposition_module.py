"""
Re-composition Module for StreamSpeech

Re-composes and improves speech units/text/waveform when sentence boundaries
are detected by CT-Transformer punctuation predictor.
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RecompositionBuffer:
    """
    Buffer management for re-composition.
    
    Stores units, text, and waveform segments until sentence boundary
    is detected, then triggers re-composition.
    """
    
    def __init__(
        self,
        max_buffer_size: int = 1000,
        enable_deduplication: bool = True
    ):
        """
        Initialize re-composition buffer.
        
        Args:
            max_buffer_size: Maximum buffer size for units
            enable_deduplication: Remove duplicate consecutive units
        """
        self.max_buffer_size = max_buffer_size
        self.enable_deduplication = enable_deduplication
        
        # 버퍼
        self.unit_buffer: List[int] = []
        self.text_buffer: List[str] = []
        self.wav_buffer: List[float] = []
        
        # 메타데이터
        self.unit_timestamps: List[int] = []  # 각 유닛의 타임스탬프
        self.text_positions: List[Tuple[int, int]] = []  # (start, end) 위치
        
        logger.info(
            f"RecompositionBuffer initialized "
            f"(max_size={max_buffer_size}, dedup={enable_deduplication})"
        )
    
    def add_units(
        self,
        units: List[int],
        timestamp: Optional[int] = None
    ):
        """
        Add speech units to buffer.
        
        Args:
            units: List of unit IDs
            timestamp: Current timestamp
        """
        for unit in units:
            if self.enable_deduplication:
                # 중복 제거
                if len(self.unit_buffer) > 0 and self.unit_buffer[-1] == unit:
                    continue
            
            self.unit_buffer.append(unit)
            
            if timestamp is not None:
                self.unit_timestamps.append(timestamp)
        
        # 버퍼 크기 제한
        if len(self.unit_buffer) > self.max_buffer_size:
            overflow = len(self.unit_buffer) - self.max_buffer_size
            self.unit_buffer = self.unit_buffer[overflow:]
            self.unit_timestamps = self.unit_timestamps[overflow:]
    
    def add_text(self, text: str):
        """Add text to buffer."""
        if text and text.strip():
            start_pos = len("".join(self.text_buffer))
            self.text_buffer.append(text)
            end_pos = start_pos + len(text)
            self.text_positions.append((start_pos, end_pos))
    
    def add_waveform(self, wav: torch.Tensor):
        """Add waveform to buffer."""
        if wav is not None and len(wav) > 0:
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            self.wav_buffer.extend(wav.tolist())
    
    def get_buffered_data(self) -> Dict:
        """Get all buffered data."""
        return {
            "units": self.unit_buffer.copy(),
            "text": "".join(self.text_buffer),
            "waveform": np.array(self.wav_buffer),
            "unit_count": len(self.unit_buffer),
            "text_length": len("".join(self.text_buffer)),
            "wav_samples": len(self.wav_buffer),
        }
    
    def clear(self):
        """Clear all buffers."""
        self.unit_buffer = []
        self.text_buffer = []
        self.wav_buffer = []
        self.unit_timestamps = []
        self.text_positions = []
        logger.debug("Buffers cleared")
    
    def is_empty(self) -> bool:
        """Check if buffers are empty."""
        return (
            len(self.unit_buffer) == 0 
            and len(self.text_buffer) == 0 
            and len(self.wav_buffer) == 0
        )


class RecompositionModule:
    """
    Re-composition module for improving speech output quality.
    
    When CT-Transformer detects a sentence boundary, this module
    re-composes the buffered units/text/waveform to improve quality.
    """
    
    def __init__(
        self,
        vocoder,
        device: str = "cuda",
        recomposition_strategy: str = "smooth_transition"
    ):
        """
        Initialize re-composition module.
        
        Args:
            vocoder: CodeHiFiGAN vocoder instance
            device: Device for processing
            recomposition_strategy: Strategy for re-composition
                - "smooth_transition": Add smooth transitions between segments
                - "re_synthesize": Re-synthesize entire sentence
                - "none": No re-composition (pass-through)
        """
        self.vocoder = vocoder
        self.device = device
        self.strategy = recomposition_strategy
        
        logger.info(
            f"RecompositionModule initialized "
            f"(strategy={recomposition_strategy}, device={device})"
        )
    
    def recompose(
        self,
        units: List[int],
        text: str,
        waveform: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Re-compose speech output for complete sentence.
        
        Args:
            units: Speech units for complete sentence
            text: Text with punctuation
            waveform: Original waveform (optional)
            metadata: Additional metadata
        
        Returns:
            Tuple of:
                - recomposed_wav: Re-composed waveform
                - info: Re-composition information
        """
        logger.info(
            f"[Re-composition] Processing sentence: '{text}' "
            f"(units={len(units)}, strategy={self.strategy})"
        )
        
        if self.strategy == "none" or not units:
            # No re-composition
            if waveform is not None:
                return torch.from_numpy(waveform), {"strategy": "none"}
            else:
                return torch.tensor([]), {"strategy": "none"}
        
        elif self.strategy == "re_synthesize":
            # Re-synthesize entire sentence
            return self._re_synthesize(units, text, metadata)
        
        elif self.strategy == "smooth_transition":
            # Add smooth transitions
            return self._smooth_transition(units, waveform, metadata)
        
        else:
            logger.warning(f"Unknown strategy: {self.strategy}, using pass-through")
            return torch.from_numpy(waveform) if waveform is not None else torch.tensor([]), {}
    
    def _re_synthesize(
        self,
        units: List[int],
        text: str,
        metadata: Optional[Dict]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Re-synthesize entire sentence with vocoder.
        
        This allows the vocoder to see the complete sentence context
        and potentially produce higher quality output.
        """
        try:
            # Prepare vocoder input
            x = {
                "code": torch.tensor(units, dtype=torch.long, device=self.device).view(1, -1),
            }
            
            # Add speaker if available
            if metadata and "speaker" in metadata:
                x["spkr"] = metadata["speaker"]
            
            # Re-synthesize
            wav, dur = self.vocoder(x, dur_prediction=True)
            
            logger.info(
                f"[Re-synthesis] Generated {len(wav)} samples "
                f"({len(wav)/16000:.2f}s @ 16kHz)"
            )
            
            return wav, {
                "strategy": "re_synthesize",
                "units": len(units),
                "samples": len(wav),
                "duration_sec": len(wav) / 16000
            }
            
        except Exception as e:
            logger.error(f"Re-synthesis failed: {e}")
            return torch.tensor([]), {"strategy": "re_synthesize", "error": str(e)}
    
    def _smooth_transition(
        self,
        units: List[int],
        waveform: Optional[np.ndarray],
        metadata: Optional[Dict]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Add smooth transitions between segments.
        
        Applies crossfading or other smoothing techniques at segment boundaries.
        """
        if waveform is None or len(waveform) == 0:
            return torch.tensor([]), {"strategy": "smooth_transition", "skipped": True}
        
        wav_tensor = torch.from_numpy(waveform) if isinstance(waveform, np.ndarray) else waveform
        
        # TODO: Implement crossfading or smoothing
        # For now, just return the original waveform
        
        return wav_tensor, {
            "strategy": "smooth_transition",
            "samples": len(wav_tensor)
        }


class SentenceRecomposer:
    """
    Complete sentence re-composition system.
    
    Combines buffer management and re-composition strategies.
    """
    
    def __init__(
        self,
        vocoder,
        device: str = "cuda",
        strategy: str = "re_synthesize",
        max_buffer_size: int = 1000
    ):
        """
        Initialize sentence re-composer.
        
        Args:
            vocoder: CodeHiFiGAN vocoder instance
            device: Device for processing
            strategy: Re-composition strategy
            max_buffer_size: Maximum buffer size
        """
        self.buffer = RecompositionBuffer(max_buffer_size=max_buffer_size)
        self.recomposition = RecompositionModule(
            vocoder=vocoder,
            device=device,
            recomposition_strategy=strategy
        )
        
        # 통계
        self.total_sentences = 0
        self.total_recompositions = 0
        
        logger.info("SentenceRecomposer initialized")
    
    def add_output(
        self,
        units: List[int],
        text: str,
        wav: Optional[torch.Tensor] = None,
        timestamp: Optional[int] = None
    ):
        """
        Add new output to buffer.
        
        Args:
            units: Speech units
            text: Text segment
            wav: Waveform segment
            timestamp: Current timestamp
        """
        self.buffer.add_units(units, timestamp)
        self.buffer.add_text(text)
        if wav is not None:
            self.buffer.add_waveform(wav)
    
    def trigger_recomposition(
        self,
        complete_sentence: str,
        metadata: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Trigger re-composition for complete sentence.
        
        Args:
            complete_sentence: Complete sentence with punctuation
            metadata: Additional metadata
        
        Returns:
            Tuple of:
                - recomposed_wav: Re-composed waveform
                - info: Re-composition information
        """
        self.total_sentences += 1
        
        # 버퍼에서 데이터 가져오기
        buffered = self.buffer.get_buffered_data()
        
        if buffered["unit_count"] == 0:
            logger.warning("Empty buffer, skipping re-composition")
            return torch.tensor([]), {"skipped": True}
        
        # 재조합 수행
        recomposed_wav, info = self.recomposition.recompose(
            units=buffered["units"],
            text=complete_sentence,
            waveform=buffered["waveform"],
            metadata=metadata
        )
        
        # 버퍼 초기화
        self.buffer.clear()
        
        self.total_recompositions += 1
        
        # 정보 업데이트
        info.update({
            "sentence": complete_sentence,
            "total_sentences": self.total_sentences,
            "total_recompositions": self.total_recompositions
        })
        
        return recomposed_wav, info
    
    def get_stats(self) -> Dict:
        """Get re-composer statistics."""
        return {
            "total_sentences": self.total_sentences,
            "total_recompositions": self.total_recompositions,
            "buffer_state": self.buffer.get_buffered_data(),
            "buffer_empty": self.buffer.is_empty()
        }
    
    def reset(self):
        """Reset re-composer state."""
        self.buffer.clear()
        logger.debug("SentenceRecomposer reset")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Re-composition Module Test")
    print("=" * 50)
    
    # Mock vocoder
    class MockVocoder:
        def __call__(self, x, dur_prediction=False):
            units = x["code"].cpu().numpy()[0]
            # 각 유닛당 256 샘플 (16ms @ 16kHz)
            wav = torch.randn(len(units) * 256)
            dur = torch.ones(1, len(units)) * 256
            return wav, dur
    
    # Initialize
    vocoder = MockVocoder()
    recomposer = SentenceRecomposer(
        vocoder=vocoder,
        device="cpu",
        strategy="re_synthesize"
    )
    
    # Simulate streaming
    print("\n1. Adding streaming outputs to buffer...")
    recomposer.add_output(
        units=[63, 991, 162],
        text="hello",
        wav=torch.randn(768),
        timestamp=0
    )
    print(f"   Buffer: {recomposer.get_stats()['buffer_state']['unit_count']} units")
    
    recomposer.add_output(
        units=[73, 338, 359],
        text="everyone",
        wav=torch.randn(768),
        timestamp=100
    )
    print(f"   Buffer: {recomposer.get_stats()['buffer_state']['unit_count']} units")
    
    # Trigger re-composition
    print("\n2. Sentence boundary detected!")
    complete_sentence = "hello everyone."
    wav, info = recomposer.trigger_recomposition(complete_sentence)
    
    print(f"   Re-composed: {info['sentence']}")
    print(f"   Strategy: {info['strategy']}")
    print(f"   Waveform: {len(wav)} samples ({len(wav)/16000:.2f}s)")
    
    # Stats
    print(f"\n3. Final stats:")
    print(f"   {recomposer.get_stats()}")

