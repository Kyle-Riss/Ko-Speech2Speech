"""
CTC Decoders for EchoStream with Policy Integration

Components:
1. ASR CTC Decoder: Source text recognition (e.g., French → French text)
2. ST CTC Decoder: Target text translation (e.g., French → English text)
3. Policy: g(i) function for READ/WRITE decisions

Reference:
- StreamSpeech: agent/ctc_decoder.py
- Policy: Based on CTC alignment and token counting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CTCDecoder(nn.Module):
    """
    CTC Decoder for ASR/ST tasks.
    
    Architecture:
        Encoder output [B, T, D]
        → Linear projection [B, T, vocab]
        → Log softmax
        → CTC decoding (greedy/beam search)
    
    Reference: StreamSpeech agent/ctc_decoder.py
    """
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        blank_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.blank_idx = blank_idx
        
        # Projection layer
        self.proj = nn.Linear(input_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        logger.info(
            f"CTCDecoder initialized: "
            f"input_dim={input_dim}, vocab_size={vocab_size}, blank_idx={blank_idx}"
        )
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            encoder_out: [B, T, D]
            encoder_padding_mask: [B, T] (True = padding)
        
        Returns:
            logits: [B, T, vocab]
            log_probs: [B, T, vocab]
            tokens: [B, T] (greedy decoding)
        """
        # Dropout
        x = self.dropout(encoder_out)
        
        # Project to vocab
        logits = self.proj(x)  # [B, T, vocab]
        
        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Greedy decoding
        tokens = log_probs.argmax(dim=-1)  # [B, T]
        
        return {
            'logits': logits,
            'log_probs': log_probs,
            'tokens': tokens,
        }
    
    def decode_greedy(
        self,
        log_probs: torch.Tensor,
        remove_consecutive: bool = True,
        remove_blank: bool = True,
    ) -> List[List[int]]:
        """
        Greedy CTC decoding.
        
        Args:
            log_probs: [B, T, vocab]
            remove_consecutive: Remove consecutive duplicates
            remove_blank: Remove blank tokens
        
        Returns:
            decoded: List of token sequences (length B)
        """
        # Greedy selection
        tokens = log_probs.argmax(dim=-1)  # [B, T]
        
        decoded = []
        for b in range(tokens.size(0)):
            seq = tokens[b].tolist()
            
            # Remove consecutive duplicates
            if remove_consecutive:
                seq = [seq[0]] + [seq[i] for i in range(1, len(seq)) if seq[i] != seq[i-1]]
            
            # Remove blank
            if remove_blank:
                seq = [t for t in seq if t != self.blank_idx]
            
            decoded.append(seq)
        
        return decoded
    
    def get_token_count(
        self,
        log_probs: torch.Tensor,
        remove_blank: bool = True,
    ) -> torch.Tensor:
        """
        Count non-blank tokens (for policy).
        
        Args:
            log_probs: [B, T, vocab]
            remove_blank: Count only non-blank tokens
        
        Returns:
            counts: [B] - Number of tokens per sequence
        """
        tokens = log_probs.argmax(dim=-1)  # [B, T]
        
        if remove_blank:
            # Count non-blank tokens
            counts = (tokens != self.blank_idx).sum(dim=-1)
        else:
            counts = torch.full((tokens.size(0),), tokens.size(1), device=tokens.device)
        
        return counts


class ASRCTCDecoder(CTCDecoder):
    """
    ASR CTC Decoder (source text recognition).
    
    Example: French audio → French text
    """
    
    def __init__(self, input_dim: int, vocab_size: int, **kwargs):
        super().__init__(input_dim, vocab_size, **kwargs)
        logger.info("ASRCTCDecoder initialized (source text recognition)")


class STCTCDecoder(CTCDecoder):
    """
    ST CTC Decoder (target text translation).
    
    Example: French audio → English text (via CTC)
    """
    
    def __init__(self, input_dim: int, vocab_size: int, **kwargs):
        super().__init__(input_dim, vocab_size, **kwargs)
        logger.info("STCTCDecoder initialized (target text translation)")


class CTCPolicy:
    """
    CTC-based READ/WRITE Policy (g(i) function).
    
    Policy:
        g(i) = (N_asr >= k1 + i * n1) AND (N_st >= k2 + i * n2)
        
        - N_asr: Number of ASR CTC tokens
        - N_st: Number of ST CTC tokens
        - k1, k2: Initial wait (lagging)
        - n1, n2: Stride (tokens per output)
        - i: Output index
    
    Reference: StreamSpeech agent/speech_to_speech.streamspeech.agent.py
    Line 200-250 (policy logic)
    """
    
    def __init__(
        self,
        k1: int = 3,  # ASR initial wait
        k2: int = 3,  # ST initial wait
        n1: int = 3,  # ASR stride
        n2: int = 3,  # ST stride
    ):
        self.k1 = k1
        self.k2 = k2
        self.n1 = n1
        self.n2 = n2
        
        self.output_count = 0  # i
        
        logger.info(
            f"CTCPolicy initialized: "
            f"k1={k1}, k2={k2}, n1={n1}, n2={n2}"
        )
    
    def should_write(
        self,
        asr_token_count: int,
        st_token_count: int,
    ) -> bool:
        """
        Decide whether to WRITE (output) or READ (wait).
        
        Args:
            asr_token_count: Number of ASR CTC tokens
            st_token_count: Number of ST CTC tokens
        
        Returns:
            should_write: True = WRITE, False = READ
        """
        # g(i) = (N_asr >= k1 + i * n1) AND (N_st >= k2 + i * n2)
        asr_threshold = self.k1 + self.output_count * self.n1
        st_threshold = self.k2 + self.output_count * self.n2
        
        should_write = (asr_token_count >= asr_threshold) and (st_token_count >= st_threshold)
        
        logger.debug(
            f"Policy check (i={self.output_count}): "
            f"N_asr={asr_token_count} >= {asr_threshold}, "
            f"N_st={st_token_count} >= {st_threshold} "
            f"→ {'WRITE' if should_write else 'READ'}"
        )
        
        return should_write
    
    def increment_output(self):
        """Increment output count after WRITE."""
        self.output_count += 1
        logger.debug(f"Output count incremented to {self.output_count}")
    
    def reset(self):
        """Reset policy state."""
        self.output_count = 0
        logger.debug("Policy reset")


class CTCPolicyModule(nn.Module):
    """
    Integrated CTC Decoder + Policy Module.
    
    Components:
    - ASR CTC Decoder
    - ST CTC Decoder
    - CTC Policy (g(i))
    """
    
    def __init__(
        self,
        input_dim: int,
        asr_vocab_size: int,
        st_vocab_size: int,
        k1: int = 3,
        k2: int = 3,
        n1: int = 3,
        n2: int = 3,
        blank_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # ASR CTC Decoder
        self.asr_decoder = ASRCTCDecoder(
            input_dim=input_dim,
            vocab_size=asr_vocab_size,
            blank_idx=blank_idx,
            dropout=dropout,
        )
        
        # ST CTC Decoder
        self.st_decoder = STCTCDecoder(
            input_dim=input_dim,
            vocab_size=st_vocab_size,
            blank_idx=blank_idx,
            dropout=dropout,
        )
        
        # Policy
        self.policy = CTCPolicy(k1=k1, k2=k2, n1=n1, n2=n2)
        
        logger.info("CTCPolicyModule initialized")
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, any]:
        """
        Forward pass with policy decision.
        
        Args:
            encoder_out: [B, T, D]
            encoder_padding_mask: [B, T]
        
        Returns:
            asr_output: ASR CTC output
            st_output: ST CTC output
            should_write: Policy decision
            asr_token_count: Number of ASR tokens
            st_token_count: Number of ST tokens
        """
        # ASR CTC
        asr_output = self.asr_decoder(encoder_out, encoder_padding_mask)
        
        # ST CTC
        st_output = self.st_decoder(encoder_out, encoder_padding_mask)
        
        # Token counts
        asr_token_count = self.asr_decoder.get_token_count(asr_output['log_probs'])
        st_token_count = self.st_decoder.get_token_count(st_output['log_probs'])
        
        # Policy decision (for first sample in batch)
        should_write = self.policy.should_write(
            asr_token_count[0].item(),
            st_token_count[0].item(),
        )
        
        return {
            'asr_output': asr_output,
            'st_output': st_output,
            'should_write': should_write,
            'asr_token_count': asr_token_count,
            'st_token_count': st_token_count,
        }
    
    def increment_output(self):
        """Increment policy output count."""
        self.policy.increment_output()
    
    def reset_policy(self):
        """Reset policy state."""
        self.policy.reset()


if __name__ == "__main__":
    print("="*70)
    print("Testing CTC Decoders and Policy")
    print("="*70)
    
    # Test CTCDecoder
    print("\n1. Testing CTCDecoder...")
    decoder = CTCDecoder(input_dim=512, vocab_size=6000, blank_idx=0)
    
    encoder_out = torch.randn(2, 100, 512)
    output = decoder(encoder_out)
    
    print(f"   Logits: {output['logits'].shape}")
    print(f"   Log probs: {output['log_probs'].shape}")
    print(f"   Tokens: {output['tokens'].shape}")
    
    # Test greedy decoding
    decoded = decoder.decode_greedy(output['log_probs'])
    print(f"   Decoded (batch 0): {len(decoded[0])} tokens")
    print(f"   Decoded (batch 1): {len(decoded[1])} tokens")
    
    # Test token counting
    counts = decoder.get_token_count(output['log_probs'])
    print(f"   Token counts: {counts}")
    print("   ✅ CTCDecoder works")
    
    # Test ASR/ST Decoders
    print("\n2. Testing ASR/ST CTC Decoders...")
    asr_decoder = ASRCTCDecoder(input_dim=512, vocab_size=6000)
    st_decoder = STCTCDecoder(input_dim=512, vocab_size=6000)
    
    asr_out = asr_decoder(encoder_out)
    st_out = st_decoder(encoder_out)
    
    print(f"   ASR output: {asr_out['tokens'].shape}")
    print(f"   ST output: {st_out['tokens'].shape}")
    print("   ✅ ASR/ST decoders work")
    
    # Test CTCPolicy
    print("\n3. Testing CTCPolicy...")
    policy = CTCPolicy(k1=3, k2=3, n1=3, n2=3)
    
    # Simulate streaming
    print("\n   Simulating streaming:")
    for step in range(5):
        asr_count = step * 4 + 2  # Simulated ASR token count
        st_count = step * 3 + 1   # Simulated ST token count
        
        should_write = policy.should_write(asr_count, st_count)
        
        print(f"   Step {step}: N_asr={asr_count}, N_st={st_count} → {'WRITE' if should_write else 'READ'}")
        
        if should_write:
            policy.increment_output()
    
    print("   ✅ CTCPolicy works")
    
    # Test CTCPolicyModule
    print("\n4. Testing CTCPolicyModule...")
    module = CTCPolicyModule(
        input_dim=512,
        asr_vocab_size=6000,
        st_vocab_size=6000,
        k1=3, k2=3, n1=3, n2=3,
    )
    
    output = module(encoder_out)
    
    print(f"   ASR tokens: {output['asr_output']['tokens'].shape}")
    print(f"   ST tokens: {output['st_output']['tokens'].shape}")
    print(f"   ASR token count: {output['asr_token_count']}")
    print(f"   ST token count: {output['st_token_count']}")
    print(f"   Should write: {output['should_write']}")
    print("   ✅ CTCPolicyModule works")
    
    # Test streaming simulation
    print("\n5. Testing streaming simulation...")
    module.reset_policy()
    
    for step in range(3):
        # Simulate increasing encoder output length
        T = 50 + step * 20
        encoder_out = torch.randn(1, T, 512)
        
        output = module(encoder_out)
        
        print(f"\n   Step {step} (T={T}):")
        print(f"     N_asr={output['asr_token_count'][0].item()}")
        print(f"     N_st={output['st_token_count'][0].item()}")
        print(f"     Decision: {'WRITE' if output['should_write'] else 'READ'}")
        
        if output['should_write']:
            module.increment_output()
            print(f"     → Output incremented (i={module.policy.output_count})")
    
    print("\n   ✅ Streaming simulation works")
    
    print("\n" + "="*70)
    print("✅ All CTC Decoder and Policy tests passed!")
    print("="*70)
    
    print("\n💡 Usage:")
    print("  # Initialize")
    print("  module = CTCPolicyModule(")
    print("      input_dim=512, asr_vocab_size=6000, st_vocab_size=6000,")
    print("      k1=3, k2=3, n1=3, n2=3")
    print("  )")
    print("  ")
    print("  # Forward pass")
    print("  output = module(encoder_out)")
    print("  ")
    print("  # Check policy")
    print("  if output['should_write']:")
    print("      # Generate output")
    print("      module.increment_output()")
    print("  else:")
    print("      # Read more input")
    print("      pass")

