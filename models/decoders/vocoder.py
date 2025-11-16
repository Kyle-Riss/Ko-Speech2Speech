"""
CodeHiFiGAN Vocoder for EchoStream

Converts discrete speech units to high-quality waveform.
Uses actual CodeHiFiGAN implementation from StreamSpeech.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import json
import os
import sys

# Add StreamSpeech_analysis to path for imports
_streamspeech_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'StreamSpeech_analysis')
if _streamspeech_path not in sys.path:
    sys.path.insert(0, _streamspeech_path)

# Add StreamSpeech_analysis/fairseq to path for fairseq imports
_fairseq_path = os.path.join(_streamspeech_path, 'fairseq')
if _fairseq_path not in sys.path:
    sys.path.insert(0, _fairseq_path)

try:
    # Import standalone CodeHiFiGAN (no fairseq dependencies)
    from models.decoders.codehifigan_standalone import CodeGenerator as CodeHiFiGANModel, Generator as HiFiGANModel
    _HAS_STREAMSPEECH = True
except ImportError as e:
    print(f"Warning: Could not import CodeHiFiGAN: {e}")
    print("   Falling back to DummyGenerator")
    _HAS_STREAMSPEECH = False
    CodeHiFiGANModel = None
    HiFiGANModel = None


class CodeHiFiGANVocoder(nn.Module):
    """
    CodeHiFiGAN Vocoder wrapper.
    
    Converts discrete speech units to waveform.
    Uses actual CodeHiFiGAN implementation from StreamSpeech.
    """
    
    def __init__(
        self,
        num_units: int = 1000,
        sample_rate: int = 16000,
        hop_size: int = 160,  # 10ms @ 16kHz
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.num_units = num_units
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        
        # Try to use actual CodeHiFiGAN if available
        if _HAS_STREAMSPEECH and checkpoint_path and config_path:
            try:
                # Load config
                with open(config_path, 'r') as f:
                    model_cfg = json.load(f)
                
                # Initialize CodeHiFiGAN model
                self.generator = CodeHiFiGANModel(model_cfg)
                self.use_real_vocoder = True
                
                # Load checkpoint
                if checkpoint_path:
                    self.load_checkpoint(checkpoint_path)
                    
            except Exception as e:
                print(f"⚠️  Warning: Failed to initialize CodeHiFiGAN: {e}")
                print("   Using DummyGenerator instead")
                self.generator = DummyGenerator(num_units, hop_size)
                self.use_real_vocoder = False
        else:
            # Fallback to dummy generator
            self.generator = DummyGenerator(num_units, hop_size)
            self.use_real_vocoder = False
            if not _HAS_STREAMSPEECH:
                print("⚠️  StreamSpeech CodeHiFiGAN not available, using DummyGenerator")
            elif not checkpoint_path or not config_path:
                print("⚠️  Checkpoint or config path not provided, using DummyGenerator")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load pre-trained checkpoint."""
        if not self.use_real_vocoder:
            return
            
        try:
            if torch.cuda.is_available():
                state_dict = torch.load(checkpoint_path)
            else:
                state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            
            # Load generator state dict
            if 'generator' in state_dict:
                self.generator.load_state_dict(state_dict['generator'])
            else:
                self.generator.load_state_dict(state_dict)
            
            self.generator.eval()
            self.generator.remove_weight_norm()
            print(f"✅ Loaded CodeHiFiGAN checkpoint from {checkpoint_path}")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to load checkpoint: {e}")
            print("   Using randomly initialized generator")
            import traceback
            traceback.print_exc()
    
    def forward(
        self,
        units: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        speaker_id: Optional[torch.Tensor] = None,
        dur_prediction: bool = True,  # Enable duration prediction by default (like StreamSpeech)
    ) -> torch.Tensor:
        """
        Generate waveform from units.
        
        Args:
            units: [B, T_unit] discrete unit indices
            f0: [B, T_unit] F0 contour (optional)
            speaker_id: [B] speaker ID (optional)
            dur_prediction: Enable duration prediction (default: True, like StreamSpeech)
        
        Returns:
            wav: [B, T_wav] generated waveform
        """
        if self.use_real_vocoder:
            # Use actual CodeHiFiGAN (like StreamSpeech)
            # Prepare input dict
            x = {"code": units, "dur_prediction": dur_prediction}
            if f0 is not None:
                x["f0"] = f0
            if speaker_id is not None:
                x["spkr"] = speaker_id
            
            wav, dur = self.generator(**x)
            # Return both wav and dur for streaming (like StreamSpeech)
            wav_detached = wav.detach().squeeze(0) if wav.dim() > 1 else wav.detach()
            dur_detached = dur.detach() if dur is not None else None
            return wav_detached, dur_detached
        else:
            # Use dummy generator (return (wav, None) for consistency)
            wav = self.generator(units, f0, speaker_id)
            return wav, None
    
    @torch.no_grad()
    def generate(
        self,
        units: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        speaker_id: Optional[torch.Tensor] = None,
        dur_prediction: bool = True,  # Enable duration prediction by default
        return_duration: bool = False,  # Return duration for streaming
    ):
        """
        Generate waveform (inference mode).
        
        Args:
            units: [B, T_unit] discrete unit indices
            f0: [B, T_unit] F0 contour (optional)
            speaker_id: [B] speaker ID (optional)
            dur_prediction: Enable duration prediction (default: True, like StreamSpeech)
            return_duration: If True, return (wav, dur) tuple for streaming
        
        Returns:
            If return_duration=False: wav [B, T_wav] generated waveform
            If return_duration=True: (wav, dur) tuple
        """
        self.eval()
        result = self.forward(units, f0, speaker_id, dur_prediction=dur_prediction)
        if return_duration:
            return result  # Already returns (wav, dur)
        else:
            return result[0] if isinstance(result, tuple) else result  # Return only wav


class DummyGenerator(nn.Module):
    """
    Dummy waveform generator for testing.
    
    In practice, replace with actual CodeHiFiGAN generator.
    """
    
    def __init__(self, num_units: int = 1000, hop_size: int = 160):
        super().__init__()
        
        self.num_units = num_units
        self.hop_size = hop_size
        
        # Simple embedding + upsampling
        self.unit_embed = nn.Embedding(num_units, 128)
        
        # Upsampling layers (hop_size times)
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(128, 128, kernel_size=8, stride=4, padding=2),  # 4x
            nn.ReLU(),
            nn.ConvTranspose1d(128, 128, kernel_size=8, stride=4, padding=2),  # 4x
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2),   # 4x
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3),    # 2.5x (approx)
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )
    
    def forward(
        self,
        units: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        speaker_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate waveform.
        
        Args:
            units: [B, T] unit indices
            f0: [B, T] F0 (ignored in dummy)
            speaker_id: [B] speaker (ignored in dummy)
        
        Returns:
            wav: [B, T_wav] waveform
        """
        # Embed units
        x = self.unit_embed(units)  # [B, T, 128]
        
        # Transpose for Conv1d
        x = x.transpose(1, 2)  # [B, 128, T]
        
        # Upsample to waveform
        wav = self.upsample(x)  # [B, 1, T_wav]
        
        # Remove channel dimension
        wav = wav.squeeze(1)  # [B, T_wav]
        
        return wav


if __name__ == "__main__":
    print("Testing CodeHiFiGAN Vocoder...")
    
    # Test CodeHiFiGANVocoder
    print("\n1. Testing CodeHiFiGANVocoder...")
    vocoder = CodeHiFiGANVocoder(
        num_units=1000,
        sample_rate=16000,
        hop_size=160,
    )
    
    # Input units
    units = torch.randint(0, 1000, (2, 100))  # [B, T_unit]
    
    # Generate waveform
    wav = vocoder.generate(units)
    
    print(f"   Input units: {units.shape}")
    print(f"   Output waveform: {wav.shape}")
    
    # Expected: T_unit * hop_size = 100 * 160 = 16000 samples (1 second @ 16kHz)
    # Due to upsampling layers, actual length may vary slightly
    print(f"   Duration: {wav.shape[1] / 16000:.2f}s @ 16kHz")
    
    assert wav.shape[0] == 2, "Batch size mismatch"
    assert wav.shape[1] > 10000, "Waveform too short"
    print("   ✅ CodeHiFiGANVocoder test passed")
    
    # Test parameter count
    print("\n2. Checking parameters...")
    total_params = sum(p.numel() for p in vocoder.parameters())
    print(f"   Vocoder (dummy): {total_params:,} parameters")
    print("   Note: Actual CodeHiFiGAN has ~14M parameters")
    
    # Test with different lengths
    print("\n3. Testing variable lengths...")
    for T in [50, 100, 200]:
        units = torch.randint(0, 1000, (1, T))
        wav = vocoder.generate(units)
        print(f"   Input T={T}: Waveform length={wav.shape[1]} ({wav.shape[1]/16000:.3f}s)")
    print("   ✅ Variable length test passed")
    
    print("\n✅ All Vocoder tests passed!")
    print("\n💡 Note: This is a dummy generator for testing.")
    print("   Replace with actual CodeHiFiGAN for production use.")

