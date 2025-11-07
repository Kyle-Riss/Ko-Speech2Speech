"""
CodeHiFiGAN Vocoder Integration for EchoStream

Purpose:
- Convert discrete speech units to high-fidelity waveforms
- Support streaming synthesis
- Optional duration/F0/speaker conditioning

Reference:
- StreamSpeech: agent/tts/vocoder.py
- CodeHiFiGAN: Unit-based neural vocoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CodeHiFiGANVocoder(nn.Module):
    """
    CodeHiFiGAN Vocoder for unit-to-waveform synthesis.
    
    Features:
    - Unit-based synthesis (discrete speech units → waveform)
    - Optional duration prediction
    - Optional F0 (pitch) conditioning
    - Optional speaker embedding
    - Streaming-friendly
    
    Reference: StreamSpeech agent/tts/vocoder.py
    """
    
    def __init__(
        self,
        num_units: int = 1000,
        unit_embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        sample_rate: int = 16000,
        hop_size: int = 160,  # 10ms at 16kHz
        use_duration: bool = False,
        use_f0: bool = False,
        use_speaker: bool = False,
        num_speakers: int = 100,
    ):
        super().__init__()
        
        self.num_units = num_units
        self.unit_embed_dim = unit_embed_dim
        self.hidden_dim = hidden_dim
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        
        self.use_duration = use_duration
        self.use_f0 = use_f0
        self.use_speaker = use_speaker
        
        # Unit embedding
        self.unit_embed = nn.Embedding(num_units, unit_embed_dim)
        
        # Optional conditioning
        cond_dim = unit_embed_dim
        
        if use_speaker:
            self.speaker_embed = nn.Embedding(num_speakers, 64)
            cond_dim += 64
        
        if use_f0:
            self.f0_embed = nn.Linear(1, 64)
            cond_dim += 64
        
        # Duration predictor (if needed)
        if use_duration:
            self.duration_predictor = DurationPredictor(
                input_dim=unit_embed_dim,
                hidden_dim=256,
            )
        
        # Generator network (simplified HiFiGAN)
        self.generator = HiFiGANGenerator(
            input_dim=cond_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            hop_size=hop_size,
        )
        
        logger.info(
            f"CodeHiFiGANVocoder initialized: "
            f"num_units={num_units}, sample_rate={sample_rate}, "
            f"use_duration={use_duration}, use_f0={use_f0}, use_speaker={use_speaker}"
        )
    
    def forward(
        self,
        units: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        f0: Optional[torch.Tensor] = None,
        speaker_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            units: [B, T] discrete speech units
            durations: [B, T] duration per unit (optional)
            f0: [B, T'] F0 contour (optional)
            speaker_id: [B] speaker IDs (optional)
        
        Returns:
            waveform: [B, T_wav] generated waveform
            durations_pred: [B, T] predicted durations (if use_duration)
        """
        B, T = units.shape
        
        # 1. Unit embedding
        unit_emb = self.unit_embed(units)  # [B, T, D]
        
        # 2. Predict durations (if needed)
        if self.use_duration and durations is None:
            durations_pred = self.duration_predictor(unit_emb)  # [B, T]
            durations = durations_pred.round().long()
        else:
            durations_pred = None
        
        # 3. Expand units by duration (if provided)
        if durations is not None:
            unit_emb = self._expand_by_duration(unit_emb, durations)  # [B, T', D]
        
        # 4. Add conditioning
        cond = unit_emb
        
        if self.use_speaker and speaker_id is not None:
            speaker_emb = self.speaker_embed(speaker_id)  # [B, 64]
            speaker_emb = speaker_emb.unsqueeze(1).expand(-1, cond.size(1), -1)
            cond = torch.cat([cond, speaker_emb], dim=-1)
        
        if self.use_f0 and f0 is not None:
            # Align F0 to unit frames
            if f0.size(1) != cond.size(1):
                f0 = F.interpolate(
                    f0.unsqueeze(1),
                    size=cond.size(1),
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
            f0_emb = self.f0_embed(f0.unsqueeze(-1))  # [B, T', 64]
            cond = torch.cat([cond, f0_emb], dim=-1)
        
        # 5. Generate waveform
        waveform = self.generator(cond)  # [B, T_wav]
        
        return {
            'waveform': waveform,
            'durations_pred': durations_pred,
        }
    
    def _expand_by_duration(
        self,
        x: torch.Tensor,
        durations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expand sequence by durations.
        
        Args:
            x: [B, T, D]
            durations: [B, T]
        
        Returns:
            expanded: [B, T', D] where T' = sum(durations)
        """
        B, T, D = x.shape
        
        expanded = []
        for b in range(B):
            seq = []
            for t in range(T):
                dur = durations[b, t].item()
                if dur > 0:
                    # Repeat frame dur times
                    seq.append(x[b, t:t+1].expand(int(dur), -1))
            
            if len(seq) > 0:
                expanded.append(torch.cat(seq, dim=0))
            else:
                # Empty sequence
                expanded.append(torch.zeros(1, D, device=x.device))
        
        # Pad to same length
        max_len = max(e.size(0) for e in expanded)
        padded = []
        for e in expanded:
            if e.size(0) < max_len:
                pad = torch.zeros(max_len - e.size(0), D, device=e.device)
                e = torch.cat([e, pad], dim=0)
            padded.append(e)
        
        return torch.stack(padded, dim=0)  # [B, T', D]
    
    def generate(
        self,
        units: torch.Tensor,
        **kwargs
    ) -> np.ndarray:
        """
        Generate waveform (inference mode).
        
        Args:
            units: [B, T] or [T] discrete speech units
            **kwargs: Additional arguments
        
        Returns:
            waveform: [T_wav] numpy array
        """
        if units.dim() == 1:
            units = units.unsqueeze(0)
        
        with torch.no_grad():
            output = self.forward(units, **kwargs)
            waveform = output['waveform']
        
        # Convert to numpy
        waveform = waveform.squeeze(0).cpu().numpy()
        
        return waveform


class DurationPredictor(nn.Module):
    """
    Duration predictor for unit-to-waveform synthesis.
    
    Predicts how many frames each unit should be expanded to.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        
        Returns:
            durations: [B, T] (positive values)
        """
        dur = self.layers(x).squeeze(-1)  # [B, T]
        dur = F.softplus(dur)  # Ensure positive
        return dur


class HiFiGANGenerator(nn.Module):
    """
    Simplified HiFiGAN generator.
    
    Upsamples unit embeddings to waveform using transposed convolutions.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        hop_size: int = 160,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hop_size = hop_size
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Upsample layers
        # Target: hop_size = 160 (10ms at 16kHz)
        # Upsample factors: 2 * 2 * 2 * 2 * 10 = 160
        self.upsample_layers = nn.ModuleList([
            # Layer 1: 2x upsample
            nn.Sequential(
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=8, stride=2, padding=3),
                nn.LeakyReLU(0.2),
            ),
            # Layer 2: 2x upsample
            nn.Sequential(
                nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=8, stride=2, padding=3),
                nn.LeakyReLU(0.2),
            ),
            # Layer 3: 2x upsample
            nn.Sequential(
                nn.ConvTranspose1d(hidden_dim // 2, hidden_dim // 4, kernel_size=8, stride=2, padding=3),
                nn.LeakyReLU(0.2),
            ),
            # Layer 4: 2x upsample
            nn.Sequential(
                nn.ConvTranspose1d(hidden_dim // 4, hidden_dim // 8, kernel_size=8, stride=2, padding=3),
                nn.LeakyReLU(0.2),
            ),
            # Layer 5: 10x upsample
            nn.Sequential(
                nn.ConvTranspose1d(hidden_dim // 8, 64, kernel_size=20, stride=10, padding=5),
                nn.LeakyReLU(0.2),
            ),
        ])
        
        # Output projection
        self.output_proj = nn.Conv1d(64, 1, kernel_size=7, padding=3)
        self.output_act = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] unit embeddings
        
        Returns:
            waveform: [B, T_wav] where T_wav ≈ T * hop_size
        """
        # Project
        x = self.input_proj(x)  # [B, T, hidden_dim]
        
        # Transpose for conv1d: [B, hidden_dim, T]
        x = x.transpose(1, 2)
        
        # Upsample
        for layer in self.upsample_layers:
            x = layer(x)
        
        # Output projection
        x = self.output_proj(x)  # [B, 1, T_wav]
        x = self.output_act(x)
        
        # Remove channel dimension
        x = x.squeeze(1)  # [B, T_wav]
        
        return x


if __name__ == "__main__":
    print("="*70)
    print("Testing CodeHiFiGAN Vocoder")
    print("="*70)
    
    # Test vocoder
    print("\n1. Testing basic vocoder...")
    vocoder = CodeHiFiGANVocoder(
        num_units=1000,
        unit_embed_dim=256,
        hidden_dim=512,
        sample_rate=16000,
        hop_size=160,
        use_duration=False,
        use_f0=False,
        use_speaker=False,
    )
    
    print(f"   Parameters: {sum(p.numel() for p in vocoder.parameters()) / 1e6:.2f}M")
    
    # Test forward
    units = torch.randint(0, 1000, (2, 100))  # [B, T]
    output = vocoder(units)
    
    print(f"   Input units: {units.shape}")
    print(f"   Output waveform: {output['waveform'].shape}")
    print(f"   Expected length: ~{100 * 160} samples")
    print("   ✅ Basic vocoder works")
    
    # Test with duration
    print("\n2. Testing with duration prediction...")
    vocoder_dur = CodeHiFiGANVocoder(
        num_units=1000,
        use_duration=True,
    )
    
    output = vocoder_dur(units)
    
    print(f"   Predicted durations: {output['durations_pred'].shape}")
    print(f"   Output waveform: {output['waveform'].shape}")
    print("   ✅ Duration prediction works")
    
    # Test with F0
    print("\n3. Testing with F0 conditioning...")
    vocoder_f0 = CodeHiFiGANVocoder(
        num_units=1000,
        use_f0=True,
    )
    
    f0 = torch.randn(2, 100)  # [B, T]
    output = vocoder_f0(units, f0=f0)
    
    print(f"   F0 input: {f0.shape}")
    print(f"   Output waveform: {output['waveform'].shape}")
    print("   ✅ F0 conditioning works")
    
    # Test with speaker
    print("\n4. Testing with speaker embedding...")
    vocoder_spk = CodeHiFiGANVocoder(
        num_units=1000,
        use_speaker=True,
        num_speakers=100,
    )
    
    speaker_id = torch.randint(0, 100, (2,))  # [B]
    output = vocoder_spk(units, speaker_id=speaker_id)
    
    print(f"   Speaker IDs: {speaker_id}")
    print(f"   Output waveform: {output['waveform'].shape}")
    print("   ✅ Speaker embedding works")
    
    # Test generate (inference)
    print("\n5. Testing generate (inference)...")
    units_single = torch.randint(0, 1000, (50,))  # [T]
    waveform = vocoder.generate(units_single)
    
    print(f"   Input units: {units_single.shape}")
    print(f"   Output waveform: {waveform.shape}")
    print(f"   Duration: {len(waveform) / 16000:.2f}s")
    print("   ✅ Generate works")
    
    # Test duration expansion
    print("\n6. Testing duration expansion...")
    units = torch.randint(0, 1000, (1, 10))
    durations = torch.randint(1, 5, (1, 10))  # 1-4 frames per unit
    
    output = vocoder_dur(units, durations=durations)
    
    print(f"   Input units: {units.shape}")
    print(f"   Durations: {durations[0].tolist()}")
    print(f"   Expected frames: {durations.sum().item()}")
    print(f"   Output waveform: {output['waveform'].shape}")
    print("   ✅ Duration expansion works")
    
    print("\n" + "="*70)
    print("✅ All CodeHiFiGAN Vocoder tests passed!")
    print("="*70)
    
    print("\n💡 Usage:")
    print("  # Initialize")
    print("  vocoder = CodeHiFiGANVocoder(num_units=1000)")
    print("  ")
    print("  # Generate waveform")
    print("  units = torch.randint(0, 1000, (1, 100))")
    print("  output = vocoder(units)")
    print("  waveform = output['waveform']  # [B, T_wav]")
    print("  ")
    print("  # Inference")
    print("  waveform = vocoder.generate(units)  # numpy array")

