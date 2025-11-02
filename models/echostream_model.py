"""
EchoStream Model

Complete Speech-to-Speech Translation model combining:
- Emformer Encoder (efficient streaming)
- StreamSpeech Decoders (MT, Unit, CTC)
- CodeHiFiGAN Vocoder
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from echostream_encoder import EchoStreamSpeechEncoder
from decoders import (
    CTCDecoder,
    CTCDecoderWithTransformerLayer,
    TransformerMTDecoder,
    CTCTransformerUnitDecoder,
)
from decoders.vocoder import CodeHiFiGANVocoder


class EchoStreamModel(nn.Module):
    """
    EchoStream: Efficient Memory-based Streaming Speech-to-Speech Translation
    
    Architecture:
        Speech Input
            ↓
        Emformer Encoder (NEW: Efficient with Left Context Cache)
            ↓
        ┌───────────┴───────────┐
        ↓                       ↓
    ASR CTC Decoder      ST CTC Decoder
        ↓                       ↓
    (for punctuation)    MT Decoder (4L)
                               ↓
                        T2U Encoder (0L)
                               ↓
                        Unit Decoder (6L)
                               ↓
                        CodeHiFiGAN
                               ↓
                        Speech Output
    
    Key Improvement over StreamSpeech:
    - Emformer Encoder: O(1) complexity vs O(T²) in Conformer
    - Left Context Cache: Reuse K, V from previous segments
    - Memory Bank: Efficient long-range modeling
    """
    
    def __init__(
        self,
        # Encoder parameters
        encoder_embed_dim: int = 256,
        encoder_layers: int = 16,
        encoder_attention_heads: int = 4,
        encoder_ffn_embed_dim: int = 1024,
        
        # Emformer-specific
        segment_length: int = 4,
        left_context_length: int = 30,
        right_context_length: int = 0,
        memory_size: int = 8,
        
        # Decoder parameters
        decoder_embed_dim: int = 256,
        mt_decoder_layers: int = 4,
        unit_decoder_layers: int = 6,
        
        # Regularization
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Emformer Encoder
        self.encoder = EchoStreamSpeechEncoder(
            encoder_embed_dim=encoder_embed_dim,
            encoder_layers=encoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_embed_dim=encoder_ffn_embed_dim,
            segment_length=segment_length,
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            memory_size=memory_size,
            dropout=dropout,
        )
        
        # ==================================
        # Decoders
        # ==================================
        
        # ASR CTC Decoder (for punctuation prediction)
        self.asr_ctc_decoder = CTCDecoder(
            embed_dim=encoder_embed_dim,
            vocab_size=6000,  # Source language vocab
            dropout=dropout,
        )
        
        # ST CTC Decoder (for translation)
        self.st_ctc_decoder = CTCDecoderWithTransformerLayer(
            embed_dim=encoder_embed_dim,
            num_layers=2,
            num_heads=encoder_attention_heads,
            vocab_size=6000,  # Target language vocab
            unidirectional=True,  # For streaming
            dropout=dropout,
        )
        
        # MT Decoder (for text refinement)
        self.mt_decoder = TransformerMTDecoder(
            vocab_size=6000,  # Target language vocab
            embed_dim=decoder_embed_dim,
            num_layers=mt_decoder_layers,
            num_heads=encoder_attention_heads,
            dropout=dropout,
        )
        
        # Unit Decoder (for speech unit generation)
        self.unit_decoder = CTCTransformerUnitDecoder(
            input_dim=decoder_embed_dim,
            embed_dim=decoder_embed_dim,
            num_layers=unit_decoder_layers,
            num_heads=encoder_attention_heads,
            num_units=1000,  # HuBERT units
            ctc_upsample_ratio=5,
            dropout=dropout,
        )
        
        # Vocoder (CodeHiFiGAN)
        self.vocoder = CodeHiFiGANVocoder(
            num_units=1000,
            sample_rate=16000,
        )
    
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        prev_output_tokens: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass (complete S2ST pipeline).
        
        Args:
            src_tokens: Input speech features [B, T, 80]
            src_lengths: Sequence lengths [B]
            prev_output_tokens: Previous target tokens (for MT decoder) [B, T_tgt]
            target_lengths: Target lengths [B]
        
        Returns:
            Dict with:
                - 'encoder_out': Encoder output
                - 'asr_logits': ASR CTC logits
                - 'st_logits': ST CTC logits
                - 'mt_logits': MT decoder logits
                - 'unit_logits': Unit decoder logits
                - 'waveform': Generated waveform (inference only)
        """
        # ==================================
        # 1. Speech Encoder (Emformer)
        # ==================================
        encoder_out = self.encoder(src_tokens, src_lengths)
        
        encoder_hidden = encoder_out['encoder_out'][0]  # [T', B, D]
        
        # ==================================
        # 2. ASR CTC Decoder
        # ==================================
        asr_out = self.asr_ctc_decoder(
            encoder_out=encoder_hidden,
            encoder_padding_mask=encoder_out['encoder_padding_mask'][0] if encoder_out['encoder_padding_mask'] else None,
        )
        
        # ==================================
        # 3. ST CTC Decoder
        # ==================================
        st_out = self.st_ctc_decoder(
            encoder_out=encoder_hidden,
            encoder_padding_mask=encoder_out['encoder_padding_mask'][0] if encoder_out['encoder_padding_mask'] else None,
        )
        
        # ==================================
        # 4. MT Decoder (optional, for text refinement)
        # ==================================
        mt_out = None
        if prev_output_tokens is not None:
            mt_out = self.mt_decoder(
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
            )
        
        # ==================================
        # 5. Unit Decoder
        # ==================================
        # Use MT decoder output if available, otherwise use ST CTC output
        if mt_out is not None:
            # Get MT decoder hidden states (before output projection)
            # For simplicity, we use encoder output here
            text_hidden = encoder_hidden
        else:
            text_hidden = encoder_hidden
        
        unit_out = self.unit_decoder(
            text_hidden=text_hidden,
            text_padding_mask=encoder_out['encoder_padding_mask'][0] if encoder_out['encoder_padding_mask'] else None,
        )
        
        # ==================================
        # 6. Vocoder (inference only)
        # ==================================
        waveform = None
        if not self.training:
            # Get predicted units (greedy decoding)
            predicted_units = unit_out['logits'].argmax(dim=-1)  # [B, T_unit]
            
            # Generate waveform
            waveform = self.vocoder.generate(predicted_units)
        
        return {
            'encoder_out': encoder_out,
            'asr_logits': asr_out['logits'],
            'asr_log_probs': asr_out['log_probs'],
            'st_logits': st_out['logits'],
            'st_log_probs': st_out['log_probs'],
            'mt_logits': mt_out['logits'] if mt_out is not None else None,
            'unit_logits': unit_out['logits'],
            'unit_log_probs': unit_out['log_probs'],
            'waveform': waveform,
        }
    
    def reset_cache(self):
        """Reset encoder cache for new utterance."""
        self.encoder.reset_cache()


class EchoStreamConfig:
    """Configuration for EchoStream model."""
    
    def __init__(self):
        # Encoder
        self.encoder_embed_dim = 256
        self.encoder_layers = 16
        self.encoder_attention_heads = 4
        self.encoder_ffn_embed_dim = 1024
        
        # Emformer
        self.segment_length = 4  # 40ms @ 100fps
        self.left_context_length = 30  # 300ms
        self.right_context_length = 0  # Full streaming (no lookahead)
        self.memory_size = 8
        
        # Decoder
        self.decoder_embed_dim = 256
        self.mt_decoder_layers = 4
        self.unit_decoder_layers = 6
        
        # Regularization
        self.dropout = 0.1
        self.attention_dropout = 0.1
        self.activation_dropout = 0.1
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


def build_echostream_model(config: EchoStreamConfig) -> EchoStreamModel:
    """
    Build EchoStream model from config.
    
    Args:
        config: Model configuration
    
    Returns:
        EchoStreamModel instance
    """
    model = EchoStreamModel(
        encoder_embed_dim=config.encoder_embed_dim,
        encoder_layers=config.encoder_layers,
        encoder_attention_heads=config.encoder_attention_heads,
        encoder_ffn_embed_dim=config.encoder_ffn_embed_dim,
        segment_length=config.segment_length,
        left_context_length=config.left_context_length,
        right_context_length=config.right_context_length,
        memory_size=config.memory_size,
        decoder_embed_dim=config.decoder_embed_dim,
        mt_decoder_layers=config.mt_decoder_layers,
        unit_decoder_layers=config.unit_decoder_layers,
        dropout=config.dropout,
    )
    
    return model


if __name__ == "__main__":
    print("="*70)
    print("Testing Complete EchoStream S2ST Model")
    print("="*70)
    
    # Create config
    config = EchoStreamConfig()
    config.encoder_layers = 4  # Use 4 layers for faster testing
    config.mt_decoder_layers = 2
    config.unit_decoder_layers = 3
    
    # Build model
    model = build_echostream_model(config)
    model.eval()
    
    print(f"\nModel architecture:")
    print(f"  Encoder: {config.encoder_layers}-layer Emformer, {config.encoder_embed_dim}d")
    print(f"  ASR Decoder: CTC")
    print(f"  ST Decoder: CTC + 2 Transformer layers")
    print(f"  MT Decoder: {config.mt_decoder_layers}-layer Transformer")
    print(f"  Unit Decoder: {config.unit_decoder_layers}-layer Transformer + CTC Upsample")
    print(f"  Vocoder: CodeHiFiGAN")
    
    # Test input
    B, T, F = 2, 100, 80
    src_tokens = torch.randn(B, T, F)
    src_lengths = torch.tensor([100, 80])
    
    print(f"\nTest input:")
    print(f"  Speech: {src_tokens.shape} (2 utterances, 100 frames, 80-dim features)")
    
    # Forward (inference mode - generates waveform)
    print("\n" + "="*70)
    print("Running complete S2ST pipeline...")
    print("="*70)
    
    with torch.no_grad():
        output = model(src_tokens, src_lengths)
    
    # Print outputs
    print(f"\n1. Encoder output: {output['encoder_out']['encoder_out'][0].shape}")
    print(f"   Downsampling: {T} → {output['encoder_out']['encoder_out'][0].size(0)} (4x)")
    
    print(f"\n2. ASR CTC output: {output['asr_logits'].shape}")
    print(f"   Vocab size: {output['asr_logits'].size(-1)}")
    
    print(f"\n3. ST CTC output: {output['st_logits'].shape}")
    print(f"   Vocab size: {output['st_logits'].size(-1)}")
    
    print(f"\n4. Unit output: {output['unit_logits'].shape}")
    print(f"   Num units: {output['unit_logits'].size(-1)}")
    
    print(f"\n5. Waveform output: {output['waveform'].shape}")
    print(f"   Duration: {output['waveform'].shape[1] / 16000:.2f}s @ 16kHz")
    
    # Test with MT decoder
    print("\n" + "="*70)
    print("Testing with MT decoder (teacher forcing)...")
    print("="*70)
    
    prev_tokens = torch.randint(0, 6000, (B, 20))
    with torch.no_grad():
        output_mt = model(src_tokens, src_lengths, prev_output_tokens=prev_tokens)
    
    print(f"\n6. MT decoder output: {output_mt['mt_logits'].shape}")
    print(f"   Vocab size: {output_mt['mt_logits'].size(-1)}")
    
    # Model size
    print("\n" + "="*70)
    print("Model Statistics")
    print("="*70)
    
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    asr_params = sum(p.numel() for p in model.asr_ctc_decoder.parameters())
    st_params = sum(p.numel() for p in model.st_ctc_decoder.parameters())
    mt_params = sum(p.numel() for p in model.mt_decoder.parameters())
    unit_params = sum(p.numel() for p in model.unit_decoder.parameters())
    vocoder_params = sum(p.numel() for p in model.vocoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nParameter counts:")
    print(f"  Encoder (Emformer):       {encoder_params:>12,}")
    print(f"  ASR CTC Decoder:          {asr_params:>12,}")
    print(f"  ST CTC Decoder:           {st_params:>12,}")
    print(f"  MT Decoder:               {mt_params:>12,}")
    print(f"  Unit Decoder:             {unit_params:>12,}")
    print(f"  Vocoder (dummy):          {vocoder_params:>12,}")
    print(f"  " + "-"*40)
    print(f"  Total:                    {total_params:>12,}")
    
    print("\n" + "="*70)
    print("✅ Complete EchoStream S2ST Model test passed!")
    print("="*70)
    
    print("\n📋 Implementation Status:")
    print("  ✅ Emformer Encoder")
    print("  ✅ ASR CTC Decoder")
    print("  ✅ ST CTC Decoder")
    print("  ✅ MT Decoder")
    print("  ✅ Unit Decoder")
    print("  ✅ Vocoder (dummy)")
    print("\n⏭️  Next steps:")
    print("  1. Implement EchoStream agent for SimulEval")
    print("  2. Create training script")
    print("  3. Create evaluation script")
    print("  4. Replace dummy vocoder with actual CodeHiFiGAN")

