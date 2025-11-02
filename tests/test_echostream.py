"""
Comprehensive tests for EchoStream model.

Tests cover:
1. Emformer Layer functionality
2. Speech Encoder with Conv2D subsampling
3. Full model integration
4. Streaming mode
5. Cache management
6. Performance benchmarks
"""

import torch
import torch.nn as nn
import time
import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from emformer_layer import EmformerEncoderLayer, EmformerEncoder
from echostream_encoder import Conv2dSubsampler, EchoStreamSpeechEncoder
from echostream_model import EchoStreamModel, EchoStreamConfig, build_echostream_model


class TestEmformerLayer:
    """Test Emformer Layer components."""
    
    @staticmethod
    def test_basic_forward():
        """Test basic forward pass."""
        print("Testing EmformerEncoderLayer forward pass...")
        
        layer = EmformerEncoderLayer(
            embed_dim=256,
            num_heads=4,
            segment_length=4,
            left_context_length=30,
            memory_size=8,
        )
        
        # Test inputs
        center = torch.randn(4, 2, 256)
        left_key = torch.randn(30, 2, 256)
        left_value = torch.randn(30, 2, 256)
        memory = torch.randn(8, 2, 256)
        
        center_out, right_out, cache = layer(
            center=center,
            left_context_key=left_key,
            left_context_value=left_value,
            memory_bank=memory,
        )
        
        assert center_out.shape == (4, 2, 256), f"Expected (4, 2, 256), got {center_out.shape}"
        assert cache['key'].shape == (4, 2, 256), "Cache key shape mismatch"
        assert cache['value'].shape == (4, 2, 256), "Cache value shape mismatch"
        assert cache['memory'].shape == (1, 2, 256), "Cache memory shape mismatch"
        
        print("✅ Basic forward pass test passed")
    
    @staticmethod
    def test_without_context():
        """Test forward pass without left context (first segment)."""
        print("Testing without left context...")
        
        layer = EmformerEncoderLayer(embed_dim=256, num_heads=4)
        
        center = torch.randn(4, 2, 256)
        center_out, _, cache = layer(center=center)
        
        assert center_out.shape == (4, 2, 256), "Output shape mismatch"
        print("✅ Without context test passed")
    
    @staticmethod
    def test_with_right_context():
        """Test forward pass with right context (lookahead)."""
        print("Testing with right context...")
        
        layer = EmformerEncoderLayer(
            embed_dim=256,
            num_heads=4,
            right_context_length=3,
        )
        
        center = torch.randn(4, 2, 256)
        right = torch.randn(3, 2, 256)
        
        center_out, right_out, cache = layer(
            center=center,
            right=right,
        )
        
        assert center_out.shape == (4, 2, 256), "Center output shape mismatch"
        assert right_out.shape == (3, 2, 256), "Right output shape mismatch"
        print("✅ With right context test passed")


class TestEmformerEncoder:
    """Test multi-layer Emformer Encoder."""
    
    @staticmethod
    def test_multilayer():
        """Test multi-layer processing."""
        print("Testing multi-layer Emformer...")
        
        encoder = EmformerEncoder(
            num_layers=4,
            embed_dim=256,
            num_heads=4,
            segment_length=4,
        )
        
        x = torch.randn(100, 2, 256)
        lengths = torch.tensor([100, 80])
        
        output = encoder(x, lengths)
        
        assert output['encoder_out'][0].shape == (100, 2, 256), "Output shape mismatch"
        assert output['encoder_padding_mask'][0].shape == (2, 100), "Padding mask shape mismatch"
        
        print("✅ Multi-layer test passed")
    
    @staticmethod
    def test_cache_reset():
        """Test cache reset functionality."""
        print("Testing cache reset...")
        
        encoder = EmformerEncoder(num_layers=2, embed_dim=256)
        
        # First utterance
        x1 = torch.randn(50, 1, 256)
        lengths1 = torch.tensor([50])
        output1 = encoder(x1, lengths1)
        
        # Check cache is populated
        assert len(encoder.left_context_cache[0]['key']) > 0, "Cache should be populated"
        
        # Reset cache
        encoder.reset_cache()
        
        # Check cache is cleared
        assert len(encoder.left_context_cache[0]['key']) == 0, "Cache should be empty"
        
        # Second utterance (should work independently)
        x2 = torch.randn(50, 1, 256)
        lengths2 = torch.tensor([50])
        output2 = encoder(x2, lengths2)
        
        assert output2['encoder_out'][0].shape == (50, 1, 256), "Output shape mismatch"
        
        print("✅ Cache reset test passed")


class TestConv2dSubsampler:
    """Test Conv2D subsampling layer."""
    
    @staticmethod
    def test_subsampling():
        """Test 4x downsampling."""
        print("Testing Conv2D subsampling...")
        
        subsample = Conv2dSubsampler(
            input_feat_per_channel=80,
            encoder_embed_dim=256,
        )
        
        # Input: [B, T, F]
        src_tokens = torch.randn(2, 100, 80)
        src_lengths = torch.tensor([100, 80])
        
        x, output_lengths = subsample(src_tokens, src_lengths)
        
        # Output: [T', B, D] where T' = T/4
        assert x.shape == (25, 2, 256), f"Expected (25, 2, 256), got {x.shape}"
        assert output_lengths[0] == 25, f"Expected length 25, got {output_lengths[0]}"
        assert output_lengths[1] == 20, f"Expected length 20, got {output_lengths[1]}"
        
        print("✅ Subsampling test passed")


class TestEchoStreamSpeechEncoder:
    """Test complete speech encoder."""
    
    @staticmethod
    def test_full_pipeline():
        """Test full speech encoding pipeline."""
        print("Testing EchoStream Speech Encoder...")
        
        encoder = EchoStreamSpeechEncoder(
            encoder_layers=4,
            encoder_embed_dim=256,
            segment_length=4,
        )
        
        # Input: [B, T, 80]
        src_tokens = torch.randn(2, 100, 80)
        src_lengths = torch.tensor([100, 80])
        
        output = encoder(src_tokens, src_lengths)
        
        # Check output format
        assert 'encoder_out' in output, "Missing encoder_out"
        assert 'encoder_padding_mask' in output, "Missing padding mask"
        
        # Check shapes
        encoder_out = output['encoder_out'][0]
        assert encoder_out.shape == (25, 2, 256), f"Expected (25, 2, 256), got {encoder_out.shape}"
        
        print("✅ Full pipeline test passed")
    
    @staticmethod
    def test_streaming():
        """Test streaming mode with chunks."""
        print("Testing streaming mode...")
        
        encoder = EchoStreamSpeechEncoder(
            encoder_layers=4,
            encoder_embed_dim=256,
        )
        
        # Reset cache
        encoder.reset_cache()
        
        # Process 3 chunks
        chunk_size = 40
        total_output_frames = 0
        
        for i in range(3):
            chunk = torch.randn(1, chunk_size, 80)
            lengths = torch.tensor([chunk_size])
            
            output = encoder(chunk, lengths)
            frames = output['encoder_out'][0].size(0)
            total_output_frames += frames
            
            print(f"  Chunk {i+1}: {chunk_size} → {frames} frames")
        
        assert total_output_frames == 30, f"Expected 30 total frames, got {total_output_frames}"
        print("✅ Streaming test passed")


class TestEchoStreamModel:
    """Test complete EchoStream model."""
    
    @staticmethod
    def test_model_creation():
        """Test model creation from config."""
        print("Testing EchoStream Model creation...")
        
        config = EchoStreamConfig()
        config.encoder_layers = 4  # Use fewer layers for testing
        
        model = build_echostream_model(config)
        
        # Check model structure
        assert hasattr(model, 'encoder'), "Missing encoder"
        assert model.encoder.emformer.num_layers == 4, "Incorrect number of layers"
        
        print("✅ Model creation test passed")
    
    @staticmethod
    def test_model_forward():
        """Test model forward pass."""
        print("Testing model forward pass...")
        
        config = EchoStreamConfig()
        config.encoder_layers = 4
        model = build_echostream_model(config)
        
        src_tokens = torch.randn(2, 100, 80)
        src_lengths = torch.tensor([100, 80])
        
        output = model(src_tokens, src_lengths)
        
        assert 'encoder_out' in output, "Missing encoder_out"
        encoder_out = output['encoder_out']['encoder_out'][0]
        assert encoder_out.shape == (25, 2, 256), f"Expected (25, 2, 256), got {encoder_out.shape}"
        
        print("✅ Model forward test passed")
    
    @staticmethod
    def test_parameter_count():
        """Test parameter count."""
        print("Testing parameter count...")
        
        config = EchoStreamConfig()
        config.encoder_layers = 16  # Full model
        model = build_echostream_model(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Encoder parameters: {encoder_params:,}")
        
        # Rough check (should be around 20M for 16-layer encoder)
        assert encoder_params > 10_000_000, "Encoder seems too small"
        assert encoder_params < 30_000_000, "Encoder seems too large"
        
        print("✅ Parameter count test passed")


class TestPerformance:
    """Performance benchmarks."""
    
    @staticmethod
    def test_inference_speed():
        """Benchmark inference speed."""
        print("\nBenchmarking inference speed...")
        
        config = EchoStreamConfig()
        config.encoder_layers = 4
        model = build_echostream_model(config)
        model.eval()
        
        # Prepare input
        src_tokens = torch.randn(1, 1000, 80)  # 10 seconds @ 100fps
        src_lengths = torch.tensor([1000])
        
        # Warmup
        with torch.no_grad():
            _ = model(src_tokens, src_lengths)
        
        # Benchmark
        num_runs = 10
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(src_tokens, src_lengths)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / num_runs
        rtf = avg_time / 10.0  # Real-time factor (10s audio)
        
        print(f"  Average inference time: {avg_time*1000:.2f}ms")
        print(f"  Real-time factor: {rtf:.4f}x")
        print(f"  Throughput: {1/avg_time:.2f} utterances/sec")
        
        assert rtf < 1.0, f"RTF should be < 1.0 (streaming), got {rtf}"
        
        print("✅ Inference speed test passed")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running EchoStream Test Suite")
    print("="*60)
    
    # Test Emformer Layer
    print("\n[1/8] Testing EmformerEncoderLayer...")
    TestEmformerLayer.test_basic_forward()
    TestEmformerLayer.test_without_context()
    TestEmformerLayer.test_with_right_context()
    
    # Test Emformer Encoder
    print("\n[2/8] Testing EmformerEncoder...")
    TestEmformerEncoder.test_multilayer()
    TestEmformerEncoder.test_cache_reset()
    
    # Test Conv2D Subsampler
    print("\n[3/8] Testing Conv2dSubsampler...")
    TestConv2dSubsampler.test_subsampling()
    
    # Test Speech Encoder
    print("\n[4/8] Testing EchoStreamSpeechEncoder...")
    TestEchoStreamSpeechEncoder.test_full_pipeline()
    TestEchoStreamSpeechEncoder.test_streaming()
    
    # Test Full Model
    print("\n[5/8] Testing EchoStreamModel...")
    TestEchoStreamModel.test_model_creation()
    TestEchoStreamModel.test_model_forward()
    
    # Test Parameters
    print("\n[6/8] Testing model parameters...")
    TestEchoStreamModel.test_parameter_count()
    
    # Performance Benchmarks
    print("\n[7/8] Running performance benchmarks...")
    TestPerformance.test_inference_speed()
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()

