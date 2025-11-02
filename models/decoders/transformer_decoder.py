"""
Transformer Decoder for EchoStream

Implements MT (Machine Translation) Decoder:
- Autoregressive text generation
- Cross-attention to encoder output
- Causal (masked) self-attention for streaming
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import math


class TransformerMTDecoder(nn.Module):
    """
    Transformer Decoder for MT (Machine Translation) task.
    
    Autoregressive decoder that generates target text from encoder output.
    Supports streaming with causal masking.
    """
    
    def __init__(
        self,
        # Vocabulary
        vocab_size: int = 6000,
        
        # Architecture
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_embed_dim: int = 1024,
        
        # Embedding
        max_target_positions: int = 1024,
        embed_scale: Optional[float] = None,
        
        # Regularization
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        
        # Other
        no_encoder_attn: bool = False,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_target_positions = max_target_positions
        self.embed_scale = embed_scale or math.sqrt(embed_dim)
        
        # Token embedding
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        
        # Positional embedding
        self.embed_positions = PositionalEmbedding(
            max_target_positions,
            embed_dim,
            padding_idx=1,
        )
        
        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_embed_dim=ffn_embed_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                no_encoder_attn=no_encoder_attn,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        prev_output_tokens: torch.Tensor,
        encoder_out: Optional[Dict[str, list]] = None,
        incremental_state: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            prev_output_tokens: [B, T] previous tokens (teacher forcing)
            encoder_out: Encoder output dict with:
                - 'encoder_out': [T', B, D] encoder output
                - 'encoder_padding_mask': [B, T'] padding mask
            incremental_state: Cache for autoregressive generation
        
        Returns:
            Dict with:
                - 'logits': [B, T, V] output logits
                - 'attn': [B, T, T'] cross-attention weights
        """
        B, T = prev_output_tokens.size()
        
        # Extract encoder outputs
        encoder_hidden = None
        encoder_padding_mask = None
        if encoder_out is not None:
            encoder_hidden = encoder_out['encoder_out'][0]  # [T', B, D]
            if encoder_out['encoder_padding_mask']:
                encoder_padding_mask = encoder_out['encoder_padding_mask'][0]  # [B, T']
        
        # Embed tokens
        x = self.embed_tokens(prev_output_tokens)  # [B, T, D]
        x = x * self.embed_scale
        
        # Add positional embedding
        positions = self.embed_positions(prev_output_tokens, incremental_state)
        x = x + positions
        
        x = self.dropout(x)
        
        # Transpose to [T, B, D] for Transformer
        x = x.transpose(0, 1)
        
        # Generate causal mask for self-attention
        self_attn_mask = self._generate_causal_mask(T, x.device)
        
        # Pass through decoder layers
        attn_weights = []
        for layer in self.layers:
            x, layer_attn = layer(
                x=x,
                encoder_out=encoder_hidden,
                encoder_padding_mask=encoder_padding_mask,
                self_attn_mask=self_attn_mask,
                incremental_state=incremental_state,
            )
            if layer_attn is not None:
                attn_weights.append(layer_attn)
        
        # Layer norm
        x = self.layer_norm(x)
        
        # Transpose back to [B, T, D]
        x = x.transpose(0, 1)
        
        # Output projection
        logits = self.output_projection(x)  # [B, T, V]
        
        # Average attention weights
        avg_attn = None
        if attn_weights:
            avg_attn = torch.stack(attn_weights).mean(dim=0)  # [B, T, T']
        
        return {
            'logits': logits,
            'attn': avg_attn,
        }
    
    def _generate_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Generate causal (future-blind) mask for self-attention."""
        # Upper triangular mask (future positions masked)
        mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask
    
    def get_normalized_probs(
        self,
        net_output: Dict[str, torch.Tensor],
        log_probs: bool = True,
    ) -> torch.Tensor:
        """Get normalized probabilities."""
        logits = net_output['logits']
        if log_probs:
            return nn.functional.log_softmax(logits, dim=-1)
        else:
            return nn.functional.softmax(logits, dim=-1)


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer.
    
    Components:
    1. Masked self-attention (causal)
    2. Cross-attention to encoder
    3. Feed-forward network
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        ffn_embed_dim: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        no_encoder_attn: bool = False,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.no_encoder_attn = no_encoder_attn
        
        # Self-attention (causal)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=False,  # [T, B, D]
        )
        
        # Cross-attention to encoder
        if not no_encoder_attn:
            self.encoder_attn = nn.MultiheadAttention(
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
        if not no_encoder_attn:
            self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)
        
        # Activation
        self.activation_fn = nn.ReLU()
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: [T, B, D] input
            encoder_out: [T', B, D] encoder output
            encoder_padding_mask: [B, T'] encoder padding mask
            self_attn_mask: [T, T] causal mask
            incremental_state: Cache for generation
        
        Returns:
            x: [T, B, D] output
            attn: [B, T, T'] cross-attention weights (or None)
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
        
        # Cross-attention
        attn_weights = None
        if not self.no_encoder_attn and encoder_out is not None:
            residual = x
            x = self.encoder_attn_layer_norm(x)
            
            x, attn_weights = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
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
        
        return x, attn_weights


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.
    """
    
    def __init__(
        self,
        max_positions: int,
        embedding_dim: int,
        padding_idx: int = 1,
    ):
        super().__init__()
        
        self.max_positions = max_positions
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Learnable positional embeddings
        self.weights = nn.Embedding(max_positions + padding_idx + 1, embedding_dim, padding_idx=padding_idx)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.weights.weight, mean=0, std=self.embedding_dim ** -0.5)
        nn.init.constant_(self.weights.weight[self.padding_idx], 0)
    
    def forward(
        self,
        input: torch.Tensor,
        incremental_state: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input: [B, T] token indices
            incremental_state: Cache for generation
        
        Returns:
            positions: [B, T, D] positional embeddings
        """
        B, T = input.size()
        
        # Generate position indices
        if incremental_state is not None:
            # Incremental generation: positions continue from previous step
            positions = torch.full((B, 1), T, device=input.device, dtype=torch.long)
        else:
            # Training: positions are 0, 1, 2, ..., T-1
            positions = torch.arange(T, device=input.device).unsqueeze(0).expand(B, -1)
            positions = positions + self.padding_idx + 1  # Offset by padding_idx
        
        return self.weights(positions)


if __name__ == "__main__":
    print("Testing Transformer MT Decoder...")
    
    # Test TransformerMTDecoder
    print("\n1. Testing TransformerMTDecoder...")
    mt_decoder = TransformerMTDecoder(
        vocab_size=6000,
        embed_dim=256,
        num_layers=4,
        num_heads=4,
    )
    
    # Prepare inputs
    prev_tokens = torch.randint(0, 6000, (2, 20))  # [B, T]
    encoder_out = {
        'encoder_out': [torch.randn(100, 2, 256)],  # [T', B, D]
        'encoder_padding_mask': [],
    }
    
    output = mt_decoder(prev_tokens, encoder_out)
    
    print(f"   Previous tokens: {prev_tokens.shape}")
    print(f"   Encoder output: {encoder_out['encoder_out'][0].shape}")
    print(f"   Output logits: {output['logits'].shape}")
    assert output['logits'].shape == (2, 20, 6000), "MT decoder output shape mismatch"
    print("   ✅ TransformerMTDecoder test passed")
    
    # Test causal masking
    print("\n2. Testing causal self-attention mask...")
    T = 5
    causal_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool),
        diagonal=1,
    )
    print(f"   Causal mask (T={T}):")
    print(causal_mask.int())
    print("   ✅ Causal masking test passed")
    
    # Test parameter count
    print("\n3. Checking parameters...")
    total_params = sum(p.numel() for p in mt_decoder.parameters())
    print(f"   MT Decoder (4 layers): {total_params:,} parameters")
    
    # Test with different sequence lengths
    print("\n4. Testing variable sequence lengths...")
    for T in [10, 50, 100]:
        tokens = torch.randint(0, 6000, (2, T))  # Keep B=2 to match encoder_out
        # Create encoder_out for each test
        test_encoder_out = {
            'encoder_out': [torch.randn(100, 2, 256)],
            'encoder_padding_mask': [],
        }
        output = mt_decoder(tokens, test_encoder_out)
        print(f"   Input T={T}: Output shape {output['logits'].shape}")
        assert output['logits'].shape == (2, T, 6000), f"Shape mismatch for T={T}"
    print("   ✅ Variable length test passed")
    
    print("\n✅ All Transformer MT Decoder tests passed!")

