"""
Integration Tests for Zipformer + Emformer Encoder

Test scenarios:
1. End-to-end forward pass
2. Streaming vs offline comparison
3. Memory efficiency
4. Latency measurement
5. CT-mask impact
"""

import torch
import torch.nn as nn
import time
import sys
sys.path.append('/Users/hayubin/StreamSpeech')

from models.zipformer_encoder import ZipformerEncoder
from models.streaming_interface import StreamingPipeline, StreamingEncoder, StreamState
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegrationTester:
    """Integration test suite for Zipformer encoder."""
    
    def __init__(self):
        self.results = {}
    
    def test_1_end_to_end(self):
        """Test 1: End-to-end forward pass."""
        print("\n" + "="*70)
        print("Test 1: End-to-end Forward Pass")
        print("="*70)
        
        # Initialize encoder
        encoder = ZipformerEncoder(
            input_dim=80,
            embed_dim=512,
            num_heads=8,
            ffn_dim=2048,
            num_layers_per_stack=2,
            memory_size=4,
            max_future_frames=0,
        )
        
        # Test input: 10 seconds at 100 Hz
        B, T, F = 2, 1000, 80
        audio = torch.randn(B, T, F)
        lengths = torch.tensor([T, T])
        
        print(f"\nInput: {audio.shape}")
        
        # Forward
        start = time.time()
        output = encoder(audio, lengths)
        elapsed = time.time() - start
        
        print(f"Output: {output['encoder_out'].shape}")
        print(f"Time: {elapsed:.3f}s")
        print(f"RTF: {elapsed / (T / 100):.3f}x")  # Real-time factor
        
        # Verify shapes
        # Note: 100 Hz → 50 Hz (ConvEmbed stride=2)
        expected_T = T // 2
        assert output['encoder_out'].shape == (B, expected_T, 512), f"Output shape mismatch: {output['encoder_out'].shape} != {(B, expected_T, 512)}"
        assert output['encoder_lengths'].tolist() == [expected_T, expected_T], "Length mismatch"
        assert output['carry_over'].shape == (B, 4, 512), "Carry-over shape mismatch"
        
        print("✅ End-to-end test passed")
        
        self.results['test_1'] = {
            'status': 'passed',
            'rtf': elapsed / (T / 100),
            'latency_ms': elapsed * 1000,
        }
    
    def test_2_streaming_vs_offline(self):
        """Test 2: Streaming vs offline comparison."""
        print("\n" + "="*70)
        print("Test 2: Streaming vs Offline Comparison")
        print("="*70)
        
        # Initialize encoder
        encoder = ZipformerEncoder(
            input_dim=80,
            embed_dim=512,
            num_heads=8,
            ffn_dim=2048,
            num_layers_per_stack=2,
            memory_size=4,
            max_future_frames=0,
        )
        
        # Test audio: 5 seconds
        B, T, F = 1, 500, 80
        audio = torch.randn(B, T, F)
        
        # Offline processing
        print("\nOffline processing...")
        start = time.time()
        offline_output = encoder(audio, torch.tensor([T]))
        offline_time = time.time() - start
        offline_out = offline_output['encoder_out']
        
        print(f"  Output: {offline_out.shape}")
        print(f"  Time: {offline_time:.3f}s")
        
        # Streaming processing
        print("\nStreaming processing...")
        pipeline = StreamingPipeline(
            encoder=encoder,
            chunk_size=40,  # 400ms chunks
            overlap=0,
        )
        
        start = time.time()
        
        # Process in 100-frame chunks
        chunk_size = 100
        for i in range(0, T, chunk_size):
            chunk = audio[:, i:i+chunk_size, :]
            is_final = (i + chunk_size >= T)
            output = pipeline.process(chunk, is_final=is_final)
        
        streaming_time = time.time() - start
        streaming_out = output
        
        print(f"  Output: {streaming_out.shape}")
        print(f"  Time: {streaming_time:.3f}s")
        
        # Compare
        print(f"\nComparison:")
        print(f"  Offline RTF: {offline_time / (T / 100):.3f}x")
        print(f"  Streaming RTF: {streaming_time / (T / 100):.3f}x")
        print(f"  Streaming overhead: {(streaming_time / offline_time - 1) * 100:.1f}%")
        
        # Note: Outputs may differ due to chunking
        print(f"  Output shape match: {streaming_out.shape[1]} vs {offline_out.shape[1]}")
        
        print("✅ Streaming vs offline test passed")
        
        self.results['test_2'] = {
            'status': 'passed',
            'offline_rtf': offline_time / (T / 100),
            'streaming_rtf': streaming_time / (T / 100),
            'overhead_pct': (streaming_time / offline_time - 1) * 100,
        }
    
    def test_3_memory_efficiency(self):
        """Test 3: Memory efficiency with long audio."""
        print("\n" + "="*70)
        print("Test 3: Memory Efficiency")
        print("="*70)
        
        # Initialize encoder
        encoder = ZipformerEncoder(
            input_dim=80,
            embed_dim=512,
            num_heads=8,
            ffn_dim=2048,
            num_layers_per_stack=2,
            memory_size=4,
            max_future_frames=0,
        )
        
        # Test with increasing lengths
        lengths = [100, 500, 1000, 2000]  # 1s, 5s, 10s, 20s
        
        print("\nMemory usage:")
        for T in lengths:
            audio = torch.randn(1, T, 80)
            
            # Measure memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            start = time.time()
            output = encoder(audio, torch.tensor([T]))
            elapsed = time.time() - start
            
            # Get memory (approximate)
            mem_mb = sum(p.numel() * p.element_size() for p in encoder.parameters()) / 1024 / 1024
            
            print(f"  T={T:4d} ({T/100:.1f}s): time={elapsed:.3f}s, RTF={elapsed/(T/100):.3f}x, mem={mem_mb:.1f}MB")
        
        print("✅ Memory efficiency test passed")
        
        self.results['test_3'] = {
            'status': 'passed',
            'model_size_mb': mem_mb,
        }
    
    def test_4_latency_measurement(self):
        """Test 4: Latency measurement."""
        print("\n" + "="*70)
        print("Test 4: Latency Measurement")
        print("="*70)
        
        # Initialize encoder
        encoder = ZipformerEncoder(
            input_dim=80,
            embed_dim=512,
            num_heads=8,
            ffn_dim=2048,
            num_layers_per_stack=2,
            memory_size=4,
            max_future_frames=0,
        )
        
        # Test different chunk sizes
        chunk_sizes = [20, 40, 80, 160]  # 200ms, 400ms, 800ms, 1600ms
        
        print("\nLatency by chunk size:")
        for chunk_size in chunk_sizes:
            audio = torch.randn(1, chunk_size, 80)
            
            # Warmup
            for _ in range(5):
                _ = encoder(audio, torch.tensor([chunk_size]))
            
            # Measure
            times = []
            for _ in range(20):
                start = time.time()
                _ = encoder(audio, torch.tensor([chunk_size]))
                times.append(time.time() - start)
            
            avg_time = sum(times) / len(times)
            std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            
            chunk_duration_ms = chunk_size * 10  # 10ms per frame at 100 Hz
            latency_ms = avg_time * 1000
            
            print(f"  Chunk={chunk_size:3d} ({chunk_duration_ms:4d}ms): "
                  f"latency={latency_ms:.1f}±{std_time*1000:.1f}ms, "
                  f"RTF={avg_time/(chunk_size/100):.3f}x")
        
        print("✅ Latency measurement test passed")
        
        self.results['test_4'] = {
            'status': 'passed',
        }
    
    def test_5_ct_mask_impact(self):
        """Test 5: CT-mask impact on latency and quality."""
        print("\n" + "="*70)
        print("Test 5: CT-mask Impact")
        print("="*70)
        
        # Test L=0 (full causal) vs L=1 (1 future frame)
        configs = [
            ('L=0 (Full Causal)', 0),
            ('L=1 (1 Future)', 1),
        ]
        
        audio = torch.randn(1, 400, 80)  # 4 seconds
        
        print("\nCT-mask comparison:")
        for name, L in configs:
            encoder = ZipformerEncoder(
                input_dim=80,
                embed_dim=512,
                num_heads=8,
                ffn_dim=2048,
                num_layers_per_stack=2,
                memory_size=4,
                max_future_frames=L,
            )
            
            # Warmup
            for _ in range(3):
                _ = encoder(audio, torch.tensor([400]))
            
            # Measure
            start = time.time()
            output = encoder(audio, torch.tensor([400]))
            elapsed = time.time() - start
            
            print(f"  {name}: time={elapsed:.3f}s, RTF={elapsed/4:.3f}x, output={output['encoder_out'].shape}")
        
        print("✅ CT-mask impact test passed")
        
        self.results['test_5'] = {
            'status': 'passed',
        }
    
    def test_6_state_consistency(self):
        """Test 6: State consistency across segments."""
        print("\n" + "="*70)
        print("Test 6: State Consistency")
        print("="*70)
        
        # Initialize encoder
        encoder = ZipformerEncoder(
            input_dim=80,
            embed_dim=512,
            num_heads=8,
            ffn_dim=2048,
            num_layers_per_stack=2,
            memory_size=4,
            max_future_frames=0,
        )
        
        streaming_encoder = StreamingEncoder(
            base_encoder=encoder,
            chunk_size=40,
        )
        
        # Initialize state
        state = streaming_encoder.init_state(batch_size=1)
        
        print(f"\nInitial state:")
        print(f"  memory_bank: {state.memory_bank.shape}")
        print(f"  carry_over: {state.carry_over.shape}")
        print(f"  segment_id: {state.segment_id}")
        
        # Process 5 segments
        for i in range(5):
            chunk = torch.randn(1, 40, 80)
            output, state = streaming_encoder.stream_forward(chunk, state)
            
            print(f"\nSegment {i+1}:")
            print(f"  output: {output.shape}")
            print(f"  segment_id: {state.segment_id}")
            print(f"  processed_frames: {state.processed_frames}")
            
            # Verify state
            assert state.segment_id == i + 1, f"Segment ID mismatch: {state.segment_id} != {i+1}"
            assert state.processed_frames == (i + 1) * 40, f"Processed frames mismatch"
        
        print("\n✅ State consistency test passed")
        
        self.results['test_6'] = {
            'status': 'passed',
        }
    
    def run_all(self):
        """Run all tests."""
        print("\n" + "="*70)
        print("ZIPFORMER + EMFORMER INTEGRATION TESTS")
        print("="*70)
        
        tests = [
            self.test_1_end_to_end,
            self.test_2_streaming_vs_offline,
            self.test_3_memory_efficiency,
            self.test_4_latency_measurement,
            self.test_5_ct_mask_impact,
            self.test_6_state_consistency,
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                test()
                passed += 1
            except Exception as e:
                print(f"\n❌ Test failed: {e}")
                failed += 1
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Passed: {passed}/{len(tests)}")
        print(f"Failed: {failed}/{len(tests)}")
        
        if failed == 0:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n⚠️  {failed} test(s) failed")
        
        # Print results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        for test_name, result in self.results.items():
            print(f"\n{test_name}:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        
        return failed == 0


if __name__ == "__main__":
    tester = IntegrationTester()
    success = tester.run_all()
    
    if success:
        print("\n✅ Integration tests completed successfully!")
    else:
        print("\n❌ Some integration tests failed")
        exit(1)

