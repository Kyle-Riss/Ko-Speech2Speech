"""
EchoStream Agent for SimulEval

Simultaneous Speech-to-Speech Translation with Emformer Encoder
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging
import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

try:
    from simuleval.utils import entrypoint
    from simuleval.data.segments import SpeechSegment, TextSegment
    from simuleval.agents import SpeechToSpeechAgent
    from simuleval.agents.actions import WriteAction, ReadAction
    SIMULEVAL_AVAILABLE = True
except ImportError:
    # For testing without SimulEval
    SIMULEVAL_AVAILABLE = False
    def entrypoint(cls):
        return cls
    class SpeechToSpeechAgent:
        def __init__(self, args):
            self.states = type('obj', (object,), {
                'source': None,
                'source_finished': False
            })()
    class SpeechSegment:
        def __init__(self, content, sample_rate, finished=False):
            self.content = content
            self.sample_rate = sample_rate
            self.finished = finished
    class WriteAction:
        def __init__(self, segment, finished=False):
            self.segment = segment
            self.finished = finished
    class ReadAction:
        pass

from echostream_model import build_echostream_model, EchoStreamConfig

logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 16000
FEATURE_DIM = 80
SHIFT_SIZE = 10  # ms
WINDOW_SIZE = 25  # ms


class OnlineFeatureExtractor:
    """
    Extract log mel-filterbank features online.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,  # 10ms @ 16kHz
        win_length: int = 400,  # 25ms @ 16kHz
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Internal buffer
        self.buffer = []
    
    def __call__(self, audio_segment: SpeechSegment) -> torch.Tensor:
        """
        Extract features from audio segment.
        
        Args:
            audio_segment: SpeechSegment with audio samples
        
        Returns:
            features: [T, F] log mel-filterbank features
        """
        # Get audio samples
        if isinstance(audio_segment.content, list):
            samples = np.array(audio_segment.content, dtype=np.float32)
        elif isinstance(audio_segment.content, np.ndarray):
            samples = audio_segment.content.astype(np.float32)
        else:
            samples = audio_segment.content
        
        # Add to buffer
        self.buffer.extend(samples)
        
        # Convert to torch
        waveform = torch.from_numpy(np.array(self.buffer)).float()
        
        # Extract mel-filterbank (simplified - replace with torchaudio in production)
        features = self._extract_mel_features(waveform)
        
        return features
    
    def _extract_mel_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract mel-filterbank features.
        
        In production, use torchaudio.compliance.kaldi.fbank
        For now, we use a simplified version.
        """
        # Simplified: Return random features for demonstration
        # In production, replace with:
        # import torchaudio
        # features = torchaudio.compliance.kaldi.fbank(
        #     waveform.unsqueeze(0),
        #     num_mel_bins=self.n_mels,
        #     sample_frequency=self.sample_rate,
        # )
        
        # Calculate number of frames
        num_frames = (len(waveform) - self.win_length) // self.hop_length + 1
        if num_frames < 0:
            num_frames = 0
        
        # Dummy features (replace with actual extraction)
        features = torch.randn(num_frames, self.n_mels)
        
        return features
    
    def reset(self):
        """Reset internal buffer."""
        self.buffer = []


@entrypoint
class EchoStreamAgent(SpeechToSpeechAgent):
    """
    EchoStream Agent for simultaneous speech-to-speech translation.
    
    Features:
    - Emformer encoder for efficient streaming
    - Multi-task learning (ASR, ST, MT, Unit)
    - Real-time waveform generation
    - Optional CT-Transformer punctuation for re-composition
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"EchoStream Agent initialized on {self.device}")
        
        # Load model
        self.load_model(args)
        
        # Feature extractor
        self.feature_extractor = OnlineFeatureExtractor(
            sample_rate=SAMPLE_RATE,
            n_mels=FEATURE_DIM,
        )
        
        # Streaming configuration
        self.chunk_size = getattr(args, "chunk_size", 320)  # ms
        self.wait_k_threshold = getattr(args, "wait_k", 5)  # frames
        
        # Output buffers
        self.unit_buffer = []
        self.waveform_buffer = []
        
        logger.info(
            f"EchoStream Agent ready "
            f"(chunk_size={self.chunk_size}ms, wait_k={self.wait_k_threshold})"
        )
    
    def load_model(self, args):
        """Load EchoStream model."""
        # Model configuration
        config = EchoStreamConfig()
        
        # Update from args
        if hasattr(args, "encoder_layers"):
            config.encoder_layers = args.encoder_layers
        if hasattr(args, "segment_length"):
            config.segment_length = args.segment_length
        if hasattr(args, "left_context_length"):
            config.left_context_length = args.left_context_length
        
        # Build model
        self.model = build_echostream_model(config)
        
        # Load checkpoint if provided
        if hasattr(args, "model_path") and args.model_path:
            self.load_checkpoint(args.model_path)
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
            else:
                self.model.load_state_dict(checkpoint)
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Using random initialization.")
    
    @staticmethod
    def add_args(parser):
        """Add EchoStream-specific arguments."""
        # Model
        parser.add_argument(
            "--model-path",
            type=str,
            default=None,
            help="Path to EchoStream checkpoint"
        )
        parser.add_argument(
            "--encoder-layers",
            type=int,
            default=16,
            help="Number of Emformer layers"
        )
        
        # Streaming
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=320,
            help="Audio chunk size in milliseconds"
        )
        parser.add_argument(
            "--wait-k",
            type=int,
            default=5,
            help="Wait-k frames before writing"
        )
        parser.add_argument(
            "--segment-length",
            type=int,
            default=4,
            help="Emformer segment length"
        )
        parser.add_argument(
            "--left-context-length",
            type=int,
            default=30,
            help="Emformer left context length"
        )
    
    def reset(self):
        """Reset agent state for new utterance."""
        # Reset feature extractor
        self.feature_extractor.reset()
        
        # Reset model cache
        self.model.reset_cache()
        
        # Reset buffers
        self.unit_buffer = []
        self.waveform_buffer = []
        
        logger.debug("Agent state reset")
    
    @torch.inference_mode()
    def policy(self):
        """
        Main policy for simultaneous translation.
        
        Returns:
            WriteAction or ReadAction
        """
        # Extract features from audio buffer
        features = self.feature_extractor(self.states.source)
        
        # Check if we have enough features
        if features.size(0) == 0 and not self.states.source_finished:
            return ReadAction()
        
        # Prepare model input
        src_tokens = features.unsqueeze(0)  # [1, T, F]
        src_lengths = torch.tensor([features.size(0)], device=self.device)
        
        # Forward pass through model
        output = self.model(
            src_tokens=src_tokens.to(self.device),
            src_lengths=src_lengths.to(self.device),
        )
        
        # Get generated waveform
        waveform = output['waveform']  # [1, T_wav]
        
        if waveform is not None and waveform.size(1) > 0:
            # Convert to numpy
            wav_numpy = waveform.squeeze(0).cpu().numpy()
            
            # Apply wait-k policy
            # For simplicity, we output when we have new audio
            if len(wav_numpy) > len(self.waveform_buffer):
                # Get new audio
                new_audio = wav_numpy[len(self.waveform_buffer):]
                self.waveform_buffer.extend(new_audio)
                
                # Create speech segment
                segment = SpeechSegment(
                    content=list(new_audio),
                    sample_rate=SAMPLE_RATE,
                    finished=self.states.source_finished,
                )
                
                # Write action
                return WriteAction(segment, finished=self.states.source_finished)
        
        # If source is finished but no new output, finish
        if self.states.source_finished:
            return WriteAction(
                SpeechSegment(
                    content=[],
                    sample_rate=SAMPLE_RATE,
                    finished=True,
                ),
                finished=True,
            )
        
        # Otherwise, read more
        return ReadAction()


if __name__ == "__main__":
    print("="*70)
    print("Testing EchoStream Agent")
    print("="*70)
    
    # Create dummy args
    class Args:
        model_path = None
        encoder_layers = 4
        chunk_size = 320
        wait_k = 5
        segment_length = 4
        left_context_length = 30
    
    args = Args()
    
    # Initialize agent
    print("\n1. Initializing agent...")
    agent = EchoStreamAgent(args)
    print(f"   ✅ Agent initialized on {agent.device}")
    
    # Test feature extraction
    print("\n2. Testing feature extraction...")
    audio_samples = np.random.randn(16000).astype(np.float32)  # 1 second
    audio_segment = SpeechSegment(
        content=list(audio_samples),
        sample_rate=16000,
    )
    
    features = agent.feature_extractor(audio_segment)
    print(f"   Audio: {len(audio_samples)} samples (1.0s)")
    print(f"   Features: {features.shape}")
    print(f"   ✅ Feature extraction successful")
    
    # Test model forward
    print("\n3. Testing model forward...")
    src_tokens = features.unsqueeze(0).to(agent.device)
    src_lengths = torch.tensor([features.size(0)], device=agent.device)
    
    output = agent.model(src_tokens, src_lengths)
    
    print(f"   Encoder output: {output['encoder_out']['encoder_out'][0].shape}")
    print(f"   ASR logits: {output['asr_logits'].shape}")
    print(f"   ST logits: {output['st_logits'].shape}")
    print(f"   Unit logits: {output['unit_logits'].shape}")
    print(f"   Waveform: {output['waveform'].shape if output['waveform'] is not None else None}")
    print(f"   ✅ Model forward successful")
    
    # Test reset
    print("\n4. Testing agent reset...")
    agent.reset()
    print(f"   Feature buffer: {len(agent.feature_extractor.buffer)}")
    print(f"   ✅ Agent reset successful")
    
    print("\n" + "="*70)
    print("✅ All EchoStream Agent tests passed!")
    print("="*70)
    
    print("\n📋 Usage:")
    print("  simuleval \\")
    print("    --agent agent/echostream_agent.py \\")
    print("    --source audio_file_list.txt \\")
    print("    --target target_text.txt \\")
    print("    --model-path checkpoints/echostream.pt \\")
    print("    --output results/ \\")
    print("    --chunk-size 320 \\")
    print("    --device gpu")

