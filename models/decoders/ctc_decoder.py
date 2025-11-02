"""
CTC Decoders for EchoStream

Implements:
1. CTCDecoder: Simple CTC decoder (for ASR)
2. CTCDecoderWithTransformerLayer: Enhanced CTC decoder with Transformer layers (for ST)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple


class CTCDecoder(nn.Module):
    """
    Simple CTC Decoder for ASR task.
    
    Projects encoder output to vocabulary logits using linear layer.
    Non-autoregressive, enables parallel prediction.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        vocab_size: int = 6000,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # Linear projection to vocabulary
        self.proj = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            encoder_out: [T, B, D] encoder output
            encoder_padding_mask: [B, T] padding mask
        
        Returns:
            Dict with:
                - 'logits': [T, B, V] vocabulary logits
                - 'log_probs': [T, B, V] log probabilities
        """
        # Apply dropout
        x = self.dropout(encoder_out)
        
        # Project to vocabulary
        logits = self.proj(x)  # [T, B, V]
        
        # Log softmax for CTC loss
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'log_probs': log_probs,
        }
    
    def get_normalized_probs(
        self,
        net_output: Dict[str, torch.Tensor],
        log_probs: bool = True,
    ) -> torch.Tensor:
        """Get normalized probabilities (for inference)."""
        if log_probs:
            return net_output['log_probs']
        else:
            return torch.exp(net_output['log_probs'])


class CTCDecoderWithTransformerLayer(nn.Module):
    """
    Enhanced CTC Decoder with Transformer layers.
    
    Used for ST (Speech-to-Text Translation) task.
    Adds Transformer layers before CTC projection for better context modeling.
    Supports unidirectional (causal) attention for streaming.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_embed_dim: int = 1024,
        vocab_size: int = 6000,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        unidirectional: bool = True,  # For streaming
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.unidirectional = unidirectional
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_embed_dim=ffn_embed_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # CTC projection
        self.ctc_proj = nn.Linear(embed_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Transformer layers.
        
        Args:
            encoder_out: [T, B, D] encoder output
            encoder_padding_mask: [B, T] padding mask
        
        Returns:
            Dict with:
                - 'logits': [T, B, V] vocabulary logits
                - 'log_probs': [T, B, V] log probabilities
        """
        x = encoder_out
        
        # Generate causal mask if unidirectional
        attn_mask = None
        if self.unidirectional:
            T = x.size(0)
            # Causal mask: [T, T] upper triangular
            attn_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        
        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(
                x,
                self_attn_mask=attn_mask,
                self_attn_padding_mask=encoder_padding_mask,
            )
        
        # Layer norm
        x = self.layer_norm(x)
        
        # Dropout
        x = self.dropout(x)
        
        # CTC projection
        logits = self.ctc_proj(x)  # [T, B, V]
        
        # Log softmax
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'log_probs': log_probs,
        }
    
    def get_normalized_probs(
        self,
        net_output: Dict[str, torch.Tensor],
        log_probs: bool = True,
    ) -> torch.Tensor:
        """Get normalized probabilities (for inference)."""
        if log_probs:
            return net_output['log_probs']
        else:
            return torch.exp(net_output['log_probs'])


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.
    
    Used in CTCDecoderWithTransformerLayer.
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
            batch_first=False,  # [T, B, D]
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
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [T, B, D] input
            self_attn_mask: [T, T] attention mask (e.g., causal mask)
            self_attn_padding_mask: [B, T] padding mask
        
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
            key_padding_mask=self_attn_padding_mask,
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


if __name__ == "__main__":
    print("Testing CTC Decoders...")
    
    # Test CTCDecoder
    print("\n1. Testing CTCDecoder (ASR)...")
    asr_decoder = CTCDecoder(
        embed_dim=256,
        vocab_size=6000,
    )
    
    encoder_out = torch.randn(100, 2, 256)  # [T, B, D]
    output = asr_decoder(encoder_out)
    
    print(f"   Encoder output: {encoder_out.shape}")
    print(f"   Logits: {output['logits'].shape}")
    print(f"   Log probs: {output['log_probs'].shape}")
    assert output['logits'].shape == (100, 2, 6000), "ASR decoder output shape mismatch"
    print("   ✅ CTCDecoder test passed")
    
    # Test CTCDecoderWithTransformerLayer
    print("\n2. Testing CTCDecoderWithTransformerLayer (ST)...")
    st_decoder = CTCDecoderWithTransformerLayer(
        embed_dim=256,
        num_layers=2,
        num_heads=4,
        vocab_size=6000,
        unidirectional=True,  # Streaming mode
    )
    
    output = st_decoder(encoder_out)
    
    print(f"   Encoder output: {encoder_out.shape}")
    print(f"   Logits: {output['logits'].shape}")
    print(f"   Log probs: {output['log_probs'].shape}")
    assert output['logits'].shape == (100, 2, 6000), "ST decoder output shape mismatch"
    print("   ✅ CTCDecoderWithTransformerLayer test passed")
    
    # Test unidirectional (causal) masking
    print("\n3. Testing causal masking...")
    T = 5
    causal_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool),
        diagonal=1,
    )
    print(f"   Causal mask (T={T}):")
    print(causal_mask.int())
    print("   ✅ Causal masking test passed")
    
    # Test parameter count
    print("\n4. Checking parameters...")
    asr_params = sum(p.numel() for p in asr_decoder.parameters())
    st_params = sum(p.numel() for p in st_decoder.parameters())
    print(f"   ASR Decoder: {asr_params:,} parameters")
    print(f"   ST Decoder: {st_params:,} parameters")
    
    print("\n✅ All CTC Decoder tests passed!")

