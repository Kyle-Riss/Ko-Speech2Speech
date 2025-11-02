"""
Unit Decoder for EchoStream

Converts text representations to discrete speech units.
Uses CTC upsampling and multi-frame prediction for length matching.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import math


class CTCTransformerUnitDecoder(nn.Module):
    """
    Unit Decoder with CTC Upsampling and Transformer layers.
    
    Converts text hidden states into discrete speech units.
    Key features:
    1. CTC Upsampling: Upsample text to match speech frame rate
    2. Multi-frame prediction: Predict multiple units per frame
    3. Autoregressive: Uses causal masking for streaming
    """
    
    def __init__(
        self,
        # Input
        input_dim: int = 256,  # From MT decoder
        
        # Architecture
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        ffn_embed_dim: int = 1024,
        
        # Units
        num_units: int = 1000,  # HuBERT units
        ctc_upsample_ratio: int = 5,  # Text → Speech length ratio
        num_frames_per_step: int = 1,  # Multi-frame prediction
        
        # Regularization
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_units = num_units
        self.ctc_upsample_ratio = ctc_upsample_ratio
        self.num_frames_per_step = num_frames_per_step
        
        # Input projection (if needed)
        if input_dim != embed_dim:
            self.input_proj = nn.Linear(input_dim, embed_dim)
        else:
            self.input_proj = None
        
        # CTC Upsampling layer
        self.ctc_upsample = CTCUpsample(
            embed_dim=embed_dim,
            upsample_ratio=ctc_upsample_ratio,
        )
        
        # Unit embedding (for autoregressive generation)
        self.embed_units = nn.Embedding(num_units + 1, embed_dim, padding_idx=0)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=5000)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            UnitDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_embed_dim=ffn_embed_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Output projection
        output_dim = num_units * num_frames_per_step
        self.output_proj = nn.Linear(embed_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        text_hidden: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
        prev_output_units: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            text_hidden: [T_text, B, D] text hidden states from MT decoder
            text_padding_mask: [B, T_text] padding mask
            prev_output_units: [B, T_unit] previous units (teacher forcing)
        
        Returns:
            Dict with:
                - 'logits': [B, T_unit, num_units] unit logits
                - 'log_probs': [B, T_unit, num_units] log probabilities
        """
        B = text_hidden.size(1)
        
        # Input projection
        if self.input_proj is not None:
            x = self.input_proj(text_hidden)  # [T_text, B, D]
        else:
            x = text_hidden
        
        # CTC Upsampling: [T_text, B, D] → [T_unit, B, D]
        x, upsample_lengths = self.ctc_upsample(x, text_padding_mask)
        T_unit = x.size(0)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [B, T_unit, D]
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # [T_unit, B, D]
        
        # Generate causal mask for autoregressive generation
        causal_mask = self._generate_causal_mask(T_unit, x.device)
        
        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(
                x=x,
                self_attn_mask=causal_mask,
            )
        
        # Layer norm
        x = self.layer_norm(x)
        
        # Transpose to [B, T_unit, D]
        x = x.transpose(0, 1)
        
        # Output projection
        logits = self.output_proj(x)  # [B, T_unit, num_units * num_frames]
        
        # Reshape for multi-frame output
        if self.num_frames_per_step > 1:
            # [B, T_unit, num_units * num_frames] → [B, T_unit * num_frames, num_units]
            B, T, _ = logits.shape
            logits = logits.view(B, T, self.num_frames_per_step, self.num_units)
            logits = logits.view(B, T * self.num_frames_per_step, self.num_units)
        
        # Log probabilities
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'log_probs': log_probs,
        }
    
    def _generate_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask


class CTCUpsample(nn.Module):
    """
    CTC-based upsampling layer.
    
    Upsamples text representations to match speech frame rate.
    Uses learnable upsampling to expand sequence length.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        upsample_ratio: int = 5,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.upsample_ratio = upsample_ratio
        
        # Learnable upsampling via transposed convolution
        self.upsample_conv = nn.ConvTranspose1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=upsample_ratio,
            stride=upsample_ratio,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [T, B, D] input
            padding_mask: [B, T] padding mask
        
        Returns:
            upsampled: [T * ratio, B, D] upsampled output
            lengths: [B] upsampled lengths
        """
        T, B, D = x.size()
        
        # Transpose for Conv1d: [T, B, D] → [B, D, T]
        x = x.permute(1, 2, 0)
        
        # Upsample
        x = self.upsample_conv(x)  # [B, D, T * ratio]
        
        # Transpose back: [B, D, T * ratio] → [T * ratio, B, D]
        x = x.permute(2, 0, 1)
        
        # Calculate upsampled lengths
        if padding_mask is not None:
            lengths = (~padding_mask).sum(dim=1)  # [B]
            lengths = lengths * self.upsample_ratio
        else:
            lengths = torch.full((B,), T * self.upsample_ratio, device=x.device)
        
        return x, lengths


class UnitDecoderLayer(nn.Module):
    """
    Single Unit Decoder Layer.
    
    Causal self-attention + feed-forward.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        ffn_embed_dim: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=False,
        )
        
        # Feed-forward
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        
        # Layer norm
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)
        
        # Activation
        self.activation_fn = nn.ReLU()
    
    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [T, B, D] input
            self_attn_mask: [T, T] causal mask
        
        Returns:
            x: [T, B, D] output
        """
        # Self-attention
        residual = x
        x = self.self_attn_layer_norm(x)
        
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=self_attn_mask,
        )
        
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward
        residual = x
        x = self.final_layer_norm(x)
        
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        x = residual + x
        
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.
        
        Args:
            x: [B, T, D] input
        
        Returns:
            x: [B, T, D] output with positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


if __name__ == "__main__":
    print("Testing Unit Decoder...")
    
    # Test CTCTransformerUnitDecoder
    print("\n1. Testing CTCTransformerUnitDecoder...")
    unit_decoder = CTCTransformerUnitDecoder(
        input_dim=256,
        embed_dim=256,
        num_layers=6,
        num_units=1000,
        ctc_upsample_ratio=5,
    )
    
    # Input from MT decoder
    text_hidden = torch.randn(20, 2, 256)  # [T_text, B, D]
    
    output = unit_decoder(text_hidden)
    
    print(f"   Text hidden: {text_hidden.shape}")
    print(f"   Unit logits: {output['logits'].shape}")
    print(f"   Log probs: {output['log_probs'].shape}")
    
    # Expected: T_text * upsample_ratio = 20 * 5 = 100
    assert output['logits'].shape[1] == 100, "Upsampling failed"
    assert output['logits'].shape[2] == 1000, "Unit vocabulary mismatch"
    print("   ✅ CTCTransformerUnitDecoder test passed")
    
    # Test CTC Upsampling
    print("\n2. Testing CTC Upsampling...")
    upsampler = CTCUpsample(embed_dim=256, upsample_ratio=5)
    
    x = torch.randn(20, 2, 256)
    upsampled, lengths = upsampler(x)
    
    print(f"   Input: {x.shape}")
    print(f"   Upsampled: {upsampled.shape}")
    print(f"   Lengths: {lengths}")
    assert upsampled.size(0) == 100, "Upsample ratio mismatch"
    print("   ✅ CTC Upsampling test passed")
    
    # Test parameter count
    print("\n3. Checking parameters...")
    total_params = sum(p.numel() for p in unit_decoder.parameters())
    print(f"   Unit Decoder (6 layers): {total_params:,} parameters")
    
    # Test with different input lengths
    print("\n4. Testing variable input lengths...")
    for T in [10, 50, 100]:
        text = torch.randn(T, 2, 256)
        output = unit_decoder(text)
        expected_T = T * 5  # upsample_ratio
        print(f"   Input T={T}: Output T={output['logits'].shape[1]} (expected {expected_T})")
        assert output['logits'].shape[1] == expected_T, f"Upsampling failed for T={T}"
    print("   ✅ Variable length test passed")
    
    print("\n✅ All Unit Decoder tests passed!")

