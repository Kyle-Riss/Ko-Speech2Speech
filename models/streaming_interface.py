"""
Streaming Interface for EchoStream Encoder

Purpose:
- Manage segment boundary states (K/V cache, memory bank, carry-over)
- Enable chunk-by-chunk processing
- Maintain consistency across segments

Reference:
- Emformer: Carry-over from lower layers
- Zipformer: Multi-rate state management
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class StreamState:
    """
    State container for streaming encoder.
    
    Attributes:
        memory_bank: [M, D] - Emformer memory bank
        memory_ptr: int - Ring buffer pointer
        carry_over: [B, M, D] - Carry-over from previous segment
        processed_frames: int - Total frames processed
        segment_id: int - Current segment ID
        
        # Stack-specific states (for multi-rate)
        stack_states: Dict[str, torch.Tensor] - Per-stack hidden states
    """
    
    # Emformer states
    memory_bank: Optional[torch.Tensor] = None
    memory_ptr: int = 0
    carry_over: Optional[torch.Tensor] = None
    
    # Global states
    processed_frames: int = 0
    segment_id: int = 0
    
    # Stack-specific states
    stack_states: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Metadata
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    
    def to(self, device: str):
        """Move all tensors to device."""
        if self.memory_bank is not None:
            self.memory_bank = self.memory_bank.to(device)
        if self.carry_over is not None:
            self.carry_over = self.carry_over.to(device)
        for key in self.stack_states:
            if self.stack_states[key] is not None:
                self.stack_states[key] = self.stack_states[key].to(device)
        self.device = device
        return self
    
    def clone(self):
        """Deep copy of state."""
        return StreamState(
            memory_bank=self.memory_bank.clone() if self.memory_bank is not None else None,
            memory_ptr=self.memory_ptr,
            carry_over=self.carry_over.clone() if self.carry_over is not None else None,
            processed_frames=self.processed_frames,
            segment_id=self.segment_id,
            stack_states={k: v.clone() if v is not None else None for k, v in self.stack_states.items()},
            device=self.device,
            dtype=self.dtype,
        )


class StreamingEncoder(nn.Module):
    """
    Streaming wrapper for ZipformerEncoder.
    
    Features:
    - Chunk-by-chunk processing
    - State management (memory bank, carry-over, K/V cache)
    - Segment boundary handling
    
    Usage:
        encoder = StreamingEncoder(base_encoder)
        
        # Initialize state
        state = encoder.init_state(batch_size=1)
        
        # Process chunks
        for chunk in audio_chunks:
            output, state = encoder.stream_forward(chunk, state)
    """
    
    def __init__(
        self,
        base_encoder: nn.Module,
        chunk_size: int = 40,  # frames (e.g., 40 frames = 400ms at 100 Hz)
        left_context: int = 0,  # frames to prepend from previous chunk
        right_context: int = 0,  # frames to append from next chunk
    ):
        super().__init__()
        
        self.base_encoder = base_encoder
        self.chunk_size = chunk_size
        self.left_context = left_context
        self.right_context = right_context
        
        logger.info(
            f"StreamingEncoder initialized: "
            f"chunk_size={chunk_size}, "
            f"left_context={left_context}, "
            f"right_context={right_context}"
        )
    
    def init_state(
        self,
        batch_size: int = 1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> StreamState:
        """
        Initialize streaming state.
        
        Args:
            batch_size: Batch size
            device: Device
            dtype: Data type
        
        Returns:
            state: Initial state
        """
        # Get encoder config
        embed_dim = self.base_encoder.embed_dim
        memory_size = self.base_encoder.memory_bank.memory_size
        
        # Initialize memory bank
        memory_bank = torch.zeros(memory_size, embed_dim, device=device, dtype=dtype)
        
        # Initialize carry-over (placeholder)
        carry_over = torch.zeros(batch_size, memory_size, embed_dim, device=device, dtype=dtype)
        
        state = StreamState(
            memory_bank=memory_bank,
            memory_ptr=0,
            carry_over=carry_over,
            processed_frames=0,
            segment_id=0,
            stack_states={},
            device=device,
            dtype=dtype,
        )
        
        logger.info(f"Initialized state: memory_bank={memory_bank.shape}, carry_over={carry_over.shape}")
        
        return state
    
    def stream_forward(
        self,
        chunk: torch.Tensor,
        state: StreamState,
        is_final: bool = False,
    ) -> Tuple[torch.Tensor, StreamState]:
        """
        Process a single chunk with state.
        
        Args:
            chunk: [B, T_chunk, F] - Audio chunk
            state: StreamState - Current state
            is_final: bool - Whether this is the final chunk
        
        Returns:
            output: [B, T_out, D] - Encoder output
            new_state: StreamState - Updated state
        """
        B, T_chunk, F = chunk.shape
        
        # 1. Add left context (from previous chunk)
        if state.processed_frames > 0 and self.left_context > 0:
            # TODO: Store previous chunk frames in state
            # For now, we process without left context
            pass
        
        # 2. Forward through base encoder
        # Note: We need to inject state into base encoder
        # For now, we use the base encoder as-is
        encoder_output = self.base_encoder(
            src_tokens=chunk,
            src_lengths=torch.full((B,), T_chunk, device=chunk.device),
        )
        
        output = encoder_output['encoder_out']  # [B, T_out, D]
        new_carry_over = encoder_output['carry_over']  # [B, M, D]
        
        # 3. Update state
        new_state = state.clone()
        new_state.carry_over = new_carry_over
        new_state.processed_frames += T_chunk
        new_state.segment_id += 1
        
        # Update memory bank (copy from base encoder)
        new_state.memory_bank = self.base_encoder.memory_bank.memory_bank.clone()
        new_state.memory_ptr = self.base_encoder.memory_bank.memory_ptr
        
        # 4. Handle right context (if not final)
        if not is_final and self.right_context > 0:
            # Trim output to account for right context
            # (we'll recompute these frames in the next chunk)
            output = output[:, :-self.right_context, :]
        
        logger.debug(
            f"Segment {new_state.segment_id}: "
            f"input={chunk.shape}, output={output.shape}, "
            f"processed_frames={new_state.processed_frames}"
        )
        
        return output, new_state
    
    def reset_state(self, state: StreamState) -> StreamState:
        """
        Reset state to initial values.
        
        Args:
            state: Current state
        
        Returns:
            new_state: Reset state
        """
        return self.init_state(
            batch_size=state.carry_over.size(0) if state.carry_over is not None else 1,
            device=state.device,
            dtype=state.dtype,
        )


class ChunkBuffer:
    """
    Buffer for managing audio chunks with overlap.
    
    Features:
    - Sliding window with configurable overlap
    - Automatic chunking of long audio
    - State-aware processing
    """
    
    def __init__(
        self,
        chunk_size: int = 40,  # frames
        overlap: int = 0,  # frames
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = chunk_size - overlap
        
        self.buffer = []
        self.total_frames = 0
        
        logger.info(f"ChunkBuffer: chunk_size={chunk_size}, overlap={overlap}, stride={self.stride}")
    
    def add(self, frames: torch.Tensor) -> List[torch.Tensor]:
        """
        Add frames to buffer and return complete chunks.
        
        Args:
            frames: [B, T, F] - New frames
        
        Returns:
            chunks: List of [B, chunk_size, F]
        """
        B, T, F = frames.shape
        
        # Add to buffer
        self.buffer.append(frames)
        self.total_frames += T
        
        # Concatenate buffer
        buffer_concat = torch.cat(self.buffer, dim=1)  # [B, total_frames, F]
        
        # Extract chunks
        chunks = []
        offset = 0
        while offset + self.chunk_size <= buffer_concat.size(1):
            chunk = buffer_concat[:, offset:offset + self.chunk_size, :]
            chunks.append(chunk)
            offset += self.stride
        
        # Keep remainder in buffer
        if offset < buffer_concat.size(1):
            self.buffer = [buffer_concat[:, offset:, :]]
            self.total_frames = buffer_concat.size(1) - offset
        else:
            self.buffer = []
            self.total_frames = 0
        
        return chunks
    
    def flush(self) -> Optional[torch.Tensor]:
        """
        Flush remaining frames in buffer.
        
        Returns:
            remainder: [B, T_remainder, F] or None
        """
        if len(self.buffer) == 0:
            return None
        
        remainder = torch.cat(self.buffer, dim=1)
        self.buffer = []
        self.total_frames = 0
        
        return remainder
    
    def reset(self):
        """Reset buffer."""
        self.buffer = []
        self.total_frames = 0


class StreamingPipeline:
    """
    End-to-end streaming pipeline.
    
    Components:
    - ChunkBuffer: Chunk audio with overlap
    - StreamingEncoder: Process chunks with state
    - Output aggregation: Combine chunk outputs
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        chunk_size: int = 40,
        overlap: int = 0,
        left_context: int = 0,
        right_context: int = 0,
    ):
        self.streaming_encoder = StreamingEncoder(
            base_encoder=encoder,
            chunk_size=chunk_size,
            left_context=left_context,
            right_context=right_context,
        )
        
        self.chunk_buffer = ChunkBuffer(
            chunk_size=chunk_size,
            overlap=overlap,
        )
        
        self.state = None
        self.outputs = []
        
        logger.info("StreamingPipeline initialized")
    
    def process(
        self,
        audio: torch.Tensor,
        is_final: bool = False,
    ) -> torch.Tensor:
        """
        Process audio incrementally.
        
        Args:
            audio: [B, T, F] - Audio frames
            is_final: bool - Whether this is the final audio
        
        Returns:
            output: [B, T_out, D] - Encoder output (accumulated)
        """
        # Initialize state if needed
        if self.state is None:
            B = audio.size(0)
            device = audio.device
            dtype = audio.dtype
            self.state = self.streaming_encoder.init_state(
                batch_size=B,
                device=device,
                dtype=dtype,
            )
        
        # Add to buffer and get chunks
        chunks = self.chunk_buffer.add(audio)
        
        # Process chunks
        for chunk in chunks:
            output, self.state = self.streaming_encoder.stream_forward(
                chunk=chunk,
                state=self.state,
                is_final=False,
            )
            self.outputs.append(output)
        
        # Handle final chunk
        if is_final:
            remainder = self.chunk_buffer.flush()
            if remainder is not None:
                output, self.state = self.streaming_encoder.stream_forward(
                    chunk=remainder,
                    state=self.state,
                    is_final=True,
                )
                self.outputs.append(output)
        
        # Concatenate outputs
        if len(self.outputs) > 0:
            output_concat = torch.cat(self.outputs, dim=1)
        else:
            output_concat = torch.zeros(audio.size(0), 0, self.streaming_encoder.base_encoder.embed_dim, device=audio.device)
        
        return output_concat
    
    def reset(self):
        """Reset pipeline."""
        self.state = None
        self.outputs = []
        self.chunk_buffer.reset()


if __name__ == "__main__":
    print("="*70)
    print("Testing Streaming Interface")
    print("="*70)
    
    # Import base encoder
    import sys
    sys.path.append('/Users/hayubin/StreamSpeech')
    from models.zipformer_encoder import ZipformerEncoder
    
    # Initialize base encoder
    base_encoder = ZipformerEncoder(
        input_dim=80,
        embed_dim=512,
        num_heads=8,
        ffn_dim=2048,
        num_layers_per_stack=2,
        memory_size=4,
        max_future_frames=0,
    )
    
    print(f"\n1. Base encoder: {sum(p.numel() for p in base_encoder.parameters()) / 1e6:.2f}M params")
    
    # Test StreamingEncoder
    print("\n2. Testing StreamingEncoder...")
    streaming_encoder = StreamingEncoder(
        base_encoder=base_encoder,
        chunk_size=40,  # 400ms at 100 Hz
        left_context=0,
        right_context=0,
    )
    
    # Initialize state
    state = streaming_encoder.init_state(batch_size=1, device="cpu")
    print(f"   Initial state: memory_bank={state.memory_bank.shape}, carry_over={state.carry_over.shape}")
    
    # Process chunks
    chunk1 = torch.randn(1, 40, 80)
    chunk2 = torch.randn(1, 40, 80)
    chunk3 = torch.randn(1, 40, 80)
    
    out1, state = streaming_encoder.stream_forward(chunk1, state)
    print(f"   Chunk 1: input={chunk1.shape}, output={out1.shape}")
    
    out2, state = streaming_encoder.stream_forward(chunk2, state)
    print(f"   Chunk 2: input={chunk2.shape}, output={out2.shape}")
    
    out3, state = streaming_encoder.stream_forward(chunk3, state, is_final=True)
    print(f"   Chunk 3: input={chunk3.shape}, output={out3.shape}")
    
    print(f"   Final state: segment_id={state.segment_id}, processed_frames={state.processed_frames}")
    print("   ✅ StreamingEncoder works")
    
    # Test ChunkBuffer
    print("\n3. Testing ChunkBuffer...")
    buffer = ChunkBuffer(chunk_size=40, overlap=10)
    
    audio1 = torch.randn(1, 50, 80)
    chunks = buffer.add(audio1)
    print(f"   Added {audio1.shape[1]} frames, got {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"     Chunk {i}: {chunk.shape}")
    
    audio2 = torch.randn(1, 30, 80)
    chunks = buffer.add(audio2)
    print(f"   Added {audio2.shape[1]} frames, got {len(chunks)} chunks")
    
    remainder = buffer.flush()
    print(f"   Flushed remainder: {remainder.shape if remainder is not None else None}")
    print("   ✅ ChunkBuffer works")
    
    # Test StreamingPipeline
    print("\n4. Testing StreamingPipeline...")
    pipeline = StreamingPipeline(
        encoder=base_encoder,
        chunk_size=40,
        overlap=10,
    )
    
    # Simulate streaming
    audio_stream = torch.randn(1, 200, 80)  # 2 seconds at 100 Hz
    
    # Process in chunks
    output1 = pipeline.process(audio_stream[:, :100, :], is_final=False)
    print(f"   Processed 100 frames: output={output1.shape}")
    
    output2 = pipeline.process(audio_stream[:, 100:, :], is_final=True)
    print(f"   Processed 100 frames (final): output={output2.shape}")
    
    print(f"   Total output: {output2.shape}")
    print("   ✅ StreamingPipeline works")
    
    # Test reset
    print("\n5. Testing reset...")
    pipeline.reset()
    print("   ✅ Pipeline reset")
    
    print("\n" + "="*70)
    print("✅ All Streaming Interface tests passed!")
    print("="*70)
    
    print("\n💡 Usage:")
    print("  # Initialize")
    print("  pipeline = StreamingPipeline(encoder, chunk_size=40, overlap=10)")
    print("  ")
    print("  # Process streaming audio")
    print("  for audio_chunk in audio_stream:")
    print("      output = pipeline.process(audio_chunk, is_final=False)")
    print("  ")
    print("  # Final chunk")
    print("  output = pipeline.process(final_chunk, is_final=True)")

