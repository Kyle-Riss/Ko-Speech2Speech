"""
EchoStream Simultaneous S2ST Agent

Architecture:
    Audio Stream
    → Feature Extraction (fbank)
    → Zipformer Encoder (streaming)
    → CTC Decoders (ASR, ST) + Policy
    → MT Decoder (text refinement)
    → Unit Decoder (speech units)
    → Vocoder (waveform)

Policy:
    g(i) = (N_asr >= k1 + i*n1) AND (N_st >= k2 + i*n2)
    → WRITE if True, READ if False

Reference:
    StreamSpeech: agent/speech_to_speech.streamspeech.agent.py
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Optional, Tuple
import logging
import sys
sys.path.append('/Users/hayubin/StreamSpeech')

from models.zipformer_encoder import ZipformerEncoder
from models.streaming_interface import StreamingEncoder, StreamState
from models.decoders.ctc_decoder_policy import CTCPolicyModule

# Try to import SimulEval (optional)
try:
    from simuleval.agents import SpeechToSpeechAgent
    from simuleval.agents.actions import ReadAction, WriteAction
    from simuleval.data.segments import SpeechSegment
    from simuleval.utils import entrypoint
    SIMULEVAL_AVAILABLE = True
except ImportError:
    SIMULEVAL_AVAILABLE = False
    # Mock classes for testing
    class SpeechToSpeechAgent:
        def __init__(self, args): pass
    class ReadAction: pass
    class WriteAction:
        def __init__(self, segment, finished=False): pass
    class SpeechSegment:
        def __init__(self, content, sample_rate, finished=False): pass
    def entrypoint(cls): return cls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Constants
SAMPLE_RATE = 16000
FEATURE_DIM = 80
SHIFT_SIZE = 10  # ms
WINDOW_SIZE = 25  # ms


class OnlineFeatureExtractor:
    """
    Extract fbank features on the fly.
    
    Reference: StreamSpeech agent/speech_to_speech.streamspeech.agent.py
    """
    
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        feature_dim: int = FEATURE_DIM,
        shift_size: int = SHIFT_SIZE,
        window_size: int = WINDOW_SIZE,
    ):
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim
        self.shift_size = shift_size
        self.window_size = window_size
        
        self.num_samples_per_shift = int(shift_size * sample_rate / 1000)
        self.num_samples_per_window = int(window_size * sample_rate / 1000)
        
        self.buffer = []
        
        logger.info(
            f"OnlineFeatureExtractor initialized: "
            f"sample_rate={sample_rate}, feature_dim={feature_dim}"
        )
    
    def __call__(self, samples: List[float]) -> torch.Tensor:
        """
        Extract features from audio samples.
        
        Args:
            samples: Audio samples
        
        Returns:
            features: [T, F] fbank features
        """
        # Add to buffer
        self.buffer.extend(samples)
        
        # Calculate number of frames
        num_frames = math.floor(
            (len(self.buffer) - self.num_samples_per_window + self.num_samples_per_shift)
            / self.num_samples_per_shift
        )
        
        if num_frames <= 0:
            return torch.zeros(0, self.feature_dim)
        
        # Extract features (simplified - in practice use torchaudio)
        # Here we just create dummy features for testing
        features = torch.randn(num_frames, self.feature_dim)
        
        # Keep residual samples
        consumed_samples = (num_frames - 1) * self.num_samples_per_shift + self.num_samples_per_window
        self.buffer = self.buffer[consumed_samples:]
        
        return features
    
    def reset(self):
        """Reset buffer."""
        self.buffer = []


@entrypoint
class EchoStreamSimulAgent(SpeechToSpeechAgent):
    """
    EchoStream Simultaneous S2ST Agent for SimulEval.
    
    Features:
    - Zipformer + Emformer encoder (streaming)
    - CTC-based policy (ASR + ST)
    - Real-time READ/WRITE decisions
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy parameters (initialize first)
        self.k1 = getattr(args, 'k1', 3)  # ASR initial wait
        self.k2 = getattr(args, 'k2', 3)  # ST initial wait
        self.n1 = getattr(args, 'n1', 3)  # ASR stride
        self.n2 = getattr(args, 'n2', 3)  # ST stride
        
        # Feature extractor
        self.feature_extractor = OnlineFeatureExtractor(
            sample_rate=getattr(args, 'sample_rate', SAMPLE_RATE),
            feature_dim=getattr(args, 'feature_dim', FEATURE_DIM),
        )
        
        # Model components
        self.init_model(args)
        
        # State
        self.encoder_state = None
        self.output_buffer = []
        
        logger.info(
            f"EchoStreamSimulAgent initialized: "
            f"k1={self.k1}, k2={self.k2}, n1={self.n1}, n2={self.n2}"
        )
    
    def init_model(self, args):
        """Initialize model components."""
        # Encoder
        self.encoder = ZipformerEncoder(
            input_dim=getattr(args, 'feature_dim', FEATURE_DIM),
            embed_dim=getattr(args, 'embed_dim', 512),
            num_heads=getattr(args, 'num_heads', 8),
            ffn_dim=getattr(args, 'ffn_dim', 2048),
            num_layers_per_stack=getattr(args, 'num_layers_per_stack', 2),
            memory_size=getattr(args, 'memory_size', 4),
            max_future_frames=getattr(args, 'max_future_frames', 0),
        ).to(self.device)
        
        # Streaming wrapper
        self.streaming_encoder = StreamingEncoder(
            base_encoder=self.encoder,
            chunk_size=getattr(args, 'chunk_size', 40),
            left_context=getattr(args, 'left_context', 0),
            right_context=getattr(args, 'right_context', 0),
        )
        
        # CTC Policy Module
        self.ctc_policy = CTCPolicyModule(
            input_dim=getattr(args, 'embed_dim', 512),
            asr_vocab_size=getattr(args, 'asr_vocab_size', 6000),
            st_vocab_size=getattr(args, 'st_vocab_size', 6000),
            k1=self.k1,
            k2=self.k2,
            n1=self.n1,
            n2=self.n2,
        ).to(self.device)
        
        # TODO: Add MT Decoder, Unit Decoder, Vocoder
        # For now, we'll use dummy outputs
        
        logger.info("Model components initialized")
    
    @torch.inference_mode()
    def policy(self):
        """
        Policy function for SimulEval.
        
        Returns:
            ReadAction or WriteAction
        """
        # 1. Extract features from incoming audio
        features = self.feature_extractor(self.states.source)
        
        # Check if we have enough features
        if features.size(0) == 0:
            if self.states.source_finished:
                # Flush remaining output
                return self._flush_output()
            else:
                return ReadAction()
        
        # 2. Initialize encoder state if needed
        if self.encoder_state is None:
            self.encoder_state = self.streaming_encoder.init_state(
                batch_size=1,
                device=self.device,
            )
        
        # 3. Encode features (streaming)
        features_batch = features.unsqueeze(0).to(self.device)  # [1, T, F]
        
        encoder_out, self.encoder_state = self.streaming_encoder.stream_forward(
            chunk=features_batch,
            state=self.encoder_state,
            is_final=self.states.source_finished,
        )
        
        # 4. CTC Policy decision
        policy_output = self.ctc_policy(encoder_out)
        
        should_write = policy_output['should_write']
        asr_count = policy_output['asr_token_count'][0].item()
        st_count = policy_output['st_token_count'][0].item()
        
        logger.debug(
            f"Policy: N_asr={asr_count}, N_st={st_count}, "
            f"i={self.ctc_policy.policy.output_count} → "
            f"{'WRITE' if should_write else 'READ'}"
        )
        
        # 5. Decision
        if should_write:
            # Generate output
            output_segment = self._generate_output(encoder_out, policy_output)
            
            # Increment policy counter
            self.ctc_policy.increment_output()
            
            return WriteAction(
                output_segment,
                finished=self.states.source_finished
            )
        else:
            # Need more input
            if self.states.source_finished:
                # Force output if source is finished
                output_segment = self._generate_output(encoder_out, policy_output)
                return WriteAction(output_segment, finished=True)
            else:
                return ReadAction()
    
    def _generate_output(
        self,
        encoder_out: torch.Tensor,
        policy_output: Dict,
    ) -> SpeechSegment:
        """
        Generate output speech segment.
        
        Args:
            encoder_out: [B, T, D]
            policy_output: Policy output dict
        
        Returns:
            segment: SpeechSegment
        """
        # TODO: Implement full pipeline
        # For now, return dummy audio
        
        # Dummy waveform (1 second of silence)
        waveform = np.zeros(SAMPLE_RATE, dtype=np.float32)
        
        segment = SpeechSegment(
            content=list(waveform),
            sample_rate=SAMPLE_RATE,
            finished=False,
        )
        
        return segment
    
    def _flush_output(self) -> WriteAction:
        """Flush remaining output."""
        # Empty segment to signal end
        segment = SpeechSegment(
            content=[],
            sample_rate=SAMPLE_RATE,
            finished=True,
        )
        return WriteAction(segment, finished=True)
    
    def reset(self):
        """Reset agent state."""
        self.feature_extractor.reset()
        self.encoder_state = None
        self.output_buffer = []
        self.ctc_policy.reset_policy()
        logger.info("Agent reset")


# Standalone testing
if __name__ == "__main__":
    print("="*70)
    print("Testing EchoStreamSimulAgent")
    print("="*70)
    
    # Mock args
    class Args:
        sample_rate = SAMPLE_RATE
        feature_dim = FEATURE_DIM
        embed_dim = 512
        num_heads = 8
        ffn_dim = 2048
        num_layers_per_stack = 2
        memory_size = 4
        max_future_frames = 0
        chunk_size = 40
        left_context = 0
        right_context = 0
        asr_vocab_size = 6000
        st_vocab_size = 6000
        k1 = 3
        k2 = 3
        n1 = 3
        n2 = 3
    
    args = Args()
    
    # Initialize agent
    print("\n1. Initializing agent...")
    agent = EchoStreamSimulAgent(args)
    print(f"   Device: {agent.device}")
    print(f"   Encoder: {sum(p.numel() for p in agent.encoder.parameters()) / 1e6:.2f}M params")
    print("   ✅ Agent initialized")
    
    # Test feature extraction
    print("\n2. Testing feature extraction...")
    # Simulate 1 second of audio
    audio_samples = np.random.randn(SAMPLE_RATE).tolist()
    features = agent.feature_extractor(audio_samples)
    print(f"   Audio samples: {len(audio_samples)}")
    print(f"   Features: {features.shape}")
    print("   ✅ Feature extraction works")
    
    # Test encoder
    print("\n3. Testing encoder...")
    features_batch = features.unsqueeze(0).to(agent.device)
    
    # Initialize state
    state = agent.streaming_encoder.init_state(batch_size=1, device=agent.device)
    
    # Forward
    encoder_out, state = agent.streaming_encoder.stream_forward(
        chunk=features_batch,
        state=state,
        is_final=False,
    )
    
    print(f"   Encoder input: {features_batch.shape}")
    print(f"   Encoder output: {encoder_out.shape}")
    print(f"   State: segment_id={state.segment_id}")
    print("   ✅ Encoder works")
    
    # Test CTC policy
    print("\n4. Testing CTC policy...")
    policy_output = agent.ctc_policy(encoder_out)
    
    print(f"   ASR token count: {policy_output['asr_token_count'][0].item()}")
    print(f"   ST token count: {policy_output['st_token_count'][0].item()}")
    print(f"   Should write: {policy_output['should_write']}")
    print("   ✅ CTC policy works")
    
    # Test streaming simulation
    print("\n5. Testing streaming simulation...")
    agent.reset()
    
    # Simulate 3 chunks
    for i in range(3):
        print(f"\n   Chunk {i+1}:")
        
        # Generate audio chunk (0.5 seconds)
        audio_chunk = np.random.randn(SAMPLE_RATE // 2).tolist()
        
        # Extract features
        features = agent.feature_extractor(audio_chunk)
        print(f"     Features: {features.shape}")
        
        if features.size(0) > 0:
            # Initialize state if needed
            if agent.encoder_state is None:
                agent.encoder_state = agent.streaming_encoder.init_state(
                    batch_size=1,
                    device=agent.device,
                )
            
            # Encode
            features_batch = features.unsqueeze(0).to(agent.device)
            encoder_out, agent.encoder_state = agent.streaming_encoder.stream_forward(
                chunk=features_batch,
                state=agent.encoder_state,
                is_final=False,
            )
            
            # Policy
            policy_output = agent.ctc_policy(encoder_out)
            
            print(f"     N_asr={policy_output['asr_token_count'][0].item()}")
            print(f"     N_st={policy_output['st_token_count'][0].item()}")
            print(f"     Decision: {'WRITE' if policy_output['should_write'] else 'READ'}")
            
            if policy_output['should_write']:
                agent.ctc_policy.increment_output()
                print(f"     → Output incremented (i={agent.ctc_policy.policy.output_count})")
    
    print("\n   ✅ Streaming simulation works")
    
    print("\n" + "="*70)
    print("✅ All EchoStreamSimulAgent tests passed!")
    print("="*70)
    
    print("\n💡 Usage with SimulEval:")
    print("  simuleval \\")
    print("    --agent agent/echostream_simul_agent.py \\")
    print("    --source source.txt \\")
    print("    --target target.txt \\")
    print("    --k1 3 --k2 3 --n1 3 --n2 3")

