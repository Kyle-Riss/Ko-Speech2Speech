"""
EchoStream Speech Encoder

Integrates Emformer with speech preprocessing (Conv2D subsampling)
for efficient streaming speech-to-speech translation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emformer_layer import EmformerEncoder


class Conv2dSubsampler(nn.Module):
    """
    Convolutional Subsampler for speech features.
    
    Uses two Conv2D layers with stride=2 to downsample by 4x.
    This is the standard preprocessing used in Conformer/Transformer speech models.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        input_feat_per_channel: int = 80,
        conv_out_channels: int = 256,
        encoder_embed_dim: int = 256,
        kernel_size: int = 3,
        stride: int = 2,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_feat_per_channel = input_feat_per_channel
        self.conv_out_channels = conv_out_channels
        self.encoder_embed_dim = encoder_embed_dim
        
        # Two Conv2D layers with stride=2 each → 4x downsampling
        self.conv1 = nn.Conv2d(
            input_channels,
            conv_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        
        self.conv2 = nn.Conv2d(
            conv_out_channels,
            conv_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        
        # Calculate output feature dimension after convolution
        # Input: [B, C=1, T, F=80]
        # After conv1: [B, 256, T/2, F/2]
        # After conv2: [B, 256, T/4, F/4]
        output_feat_dim = (input_feat_per_channel // 4) * conv_out_channels
        
        # Linear projection to encoder dimension
        self.out_proj = nn.Linear(output_feat_dim, encoder_embed_dim)
    
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src_tokens: [B, T, F] or [B, C, T, F]
            src_lengths: [B]
        
        Returns:
            x: [T', B, D] where T' = T/4
            output_lengths: [B]
        """
        # Ensure 4D input [B, C, T, F]
        if src_tokens.dim() == 3:
            src_tokens = src_tokens.unsqueeze(1)  # [B, 1, T, F]
        
        B, C, T, F = src_tokens.size()
        
        # Conv layers
        x = self.conv1(src_tokens)  # [B, 256, T/2, F/2]
        x = nn.functional.relu(x)
        
        x = self.conv2(x)  # [B, 256, T/4, F/4]
        x = nn.functional.relu(x)
        
        # Reshape: [B, 256, T', F'] → [B, T', 256*F']
        B, C_out, T_out, F_out = x.size()
        x = x.permute(0, 2, 1, 3)  # [B, T', 256, F']
        x = x.reshape(B, T_out, C_out * F_out)  # [B, T', 256*F']
        
        # Project to encoder dimension
        x = self.out_proj(x)  # [B, T', D]
        
        # Convert to [T', B, D] format
        x = x.transpose(0, 1)  # [T', B, D]
        
        # Update lengths (downsampled by 4)
        output_lengths = ((src_lengths - 1) // 4 + 1).long()
        
        return x, output_lengths


class EchoStreamSpeechEncoder(nn.Module):
    """
    EchoStream Speech Encoder.
    
    Architecture:
        Speech Input [B, T, 80]
            ↓
        Conv2D Subsampling (4x downsample)
            ↓
        [T/4, B, 256]
            ↓
        Emformer Encoder (16 layers)
            ↓
        [T/4, B, 256]
    
    Key features:
    - Efficient streaming with Left Context Cache
    - Memory Bank for long-range dependencies
    - O(1) complexity per segment (vs O(T²) in Conformer)
    """
    
    def __init__(
        self,
        # Input parameters
        input_feat_per_channel: int = 80,
        input_channels: int = 1,
        
        # Encoder parameters
        encoder_embed_dim: int = 256,
        encoder_layers: int = 16,
        encoder_attention_heads: int = 4,
        encoder_ffn_embed_dim: int = 1024,
        
        # Emformer-specific parameters
        segment_length: int = 4,
        left_context_length: int = 30,
        right_context_length: int = 0,
        memory_size: int = 8,
        
        # Regularization
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder_embed_dim = encoder_embed_dim
        self.segment_length = segment_length
        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        
        # Subsampling layer
        self.subsample = Conv2dSubsampler(
            input_channels=input_channels,
            input_feat_per_channel=input_feat_per_channel,
            conv_out_channels=256,
            encoder_embed_dim=encoder_embed_dim,
        )
        
        # Emformer encoder
        self.emformer = EmformerEncoder(
            num_layers=encoder_layers,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_attention_heads,
            segment_length=segment_length,
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            memory_size=memory_size,
            ffn_embed_dim=encoder_ffn_embed_dim,
            dropout=dropout,
        )
        
        self.dropout = dropout
    
    def reset_cache(self):
        """Reset Emformer cache for new utterance."""
        self.emformer.reset_cache()
    
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> Dict[str, list]:
        """
        Forward pass.
        
        Args:
            src_tokens: Input features [B, T, 80]
            src_lengths: Sequence lengths [B]
        
        Returns:
            Dict with:
                - 'encoder_out': List of [T', B, D]
                - 'encoder_padding_mask': List of [B, T']
                - 'encoder_embedding': Empty list
                - 'encoder_states': Empty list
                - 'src_tokens': Empty list
                - 'src_lengths': Empty list
        """
        # Subsampling: [B, T, 80] → [T/4, B, 256]
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        
        # Emformer encoding
        emformer_out = self.emformer(x, input_lengths)
        
        # Return in fairseq format
        return {
            'encoder_out': emformer_out['encoder_out'],  # [T', B, D]
            'encoder_padding_mask': emformer_out['encoder_padding_mask'],  # [B, T']
            'encoder_embedding': [],
            'encoder_states': [],
            'src_tokens': [],
            'src_lengths': [],
        }
    
    def reorder_encoder_out(self, encoder_out: Dict[str, list], new_order):
        """
        Reorder encoder output for beam search.
        
        Args:
            encoder_out: Output from forward()
            new_order: New order indices
        
        Returns:
            Reordered encoder_out
        """
        if len(encoder_out['encoder_out']) == 0:
            return encoder_out
        
        new_encoder_out = [
            encoder_out['encoder_out'][0].index_select(1, new_order)
        ]
        
        new_encoder_padding_mask = []
        if len(encoder_out['encoder_padding_mask']) > 0:
            new_encoder_padding_mask = [
                encoder_out['encoder_padding_mask'][0].index_select(0, new_order)
            ]
        
        return {
            'encoder_out': new_encoder_out,
            'encoder_padding_mask': new_encoder_padding_mask,
            'encoder_embedding': [],
            'encoder_states': [],
            'src_tokens': [],
            'src_lengths': [],
        }


if __name__ == "__main__":
    print("Testing EchoStream Speech Encoder...")
    
    # Create encoder
    encoder = EchoStreamSpeechEncoder(
        input_feat_per_channel=80,
        encoder_embed_dim=256,
        encoder_layers=4,  # Use 4 layers for faster testing
        encoder_attention_heads=4,
        segment_length=4,
        left_context_length=30,
    )
    
    # Test input
    B, T, F = 2, 100, 80
    src_tokens = torch.randn(B, T, F)
    src_lengths = torch.tensor([100, 80])
    
    # Forward
    encoder_out = encoder(src_tokens, src_lengths)
    
    print(f"Input shape: {src_tokens.shape}")
    print(f"Encoder output shape: {encoder_out['encoder_out'][0].shape}")
    print(f"Padding mask shape: {encoder_out['encoder_padding_mask'][0].shape}")
    print(f"Downsampling ratio: {T} → {encoder_out['encoder_out'][0].size(0)} (4x)")
    
    # Test cache reset
    encoder.reset_cache()
    print("\n✅ Cache reset successful")
    
    # Test streaming (multiple forward passes)
    print("\nTesting streaming mode...")
    encoder.reset_cache()
    
    chunk_size = 40
    for i in range(0, T, chunk_size):
        chunk_end = min(i + chunk_size, T)
        chunk_tokens = src_tokens[:, i:chunk_end, :]
        chunk_lengths = torch.tensor([chunk_end - i, min(chunk_end - i, 80 - i)])
        
        chunk_out = encoder(chunk_tokens, chunk_lengths)
        print(f"Chunk {i//chunk_size + 1}: Input {chunk_tokens.shape[1]} → Output {chunk_out['encoder_out'][0].size(0)}")
    
    print("\n✅ EchoStream Speech Encoder test passed!")

