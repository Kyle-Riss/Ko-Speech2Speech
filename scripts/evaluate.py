"""
EchoStream Evaluation Script

Evaluate EchoStream model for simultaneous speech-to-speech translation.
Metrics: BLEU, ASR-BLEU, Latency (AL, AP, DAL)
"""

import torch
import argparse
import logging
from pathlib import Path
import json
import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from echostream_model import build_echostream_model, EchoStreamConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class EchoStreamEvaluator:
    """
    Evaluator for EchoStream model.
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
        # Metrics
        self.total_samples = 0
        self.total_latency = 0
        self.outputs = []
    
    @torch.no_grad()
    def evaluate_sample(self, audio, reference_text=None):
        """
        Evaluate single sample.
        
        Args:
            audio: [T, F] audio features
            reference_text: Reference text (optional)
        
        Returns:
            output_dict: Dict with predictions
        """
        self.model.eval()
        
        # Prepare input
        src_tokens = audio.unsqueeze(0).to(self.device)  # [1, T, F]
        src_lengths = torch.tensor([audio.size(0)], device=self.device)
        
        # Forward
        output = self.model(src_tokens, src_lengths)
        
        # Get predictions
        # ASR
        asr_pred = output['asr_logits'].argmax(dim=-1)  # [T, 1, V] -> [T, 1]
        
        # ST
        st_pred = output['st_logits'].argmax(dim=-1)
        
        # Units
        unit_pred = output['unit_logits'].argmax(dim=-1)  # [1, T_unit]
        
        # Waveform
        waveform = output['waveform']  # [1, T_wav]
        
        return {
            'asr': asr_pred.squeeze().cpu().tolist(),
            'st': st_pred.squeeze().cpu().tolist(),
            'units': unit_pred.squeeze().cpu().tolist(),
            'waveform': waveform.squeeze().cpu().numpy() if waveform is not None else None,
        }
    
    def compute_metrics(self):
        """Compute evaluation metrics."""
        metrics = {
            'num_samples': self.total_samples,
            'avg_latency_ms': self.total_latency / max(self.total_samples, 1),
        }
        
        return metrics


def evaluate_model(args):
    """Main evaluation function."""
    logger.info("="*70)
    logger.info("EchoStream Evaluation")
    logger.info("="*70)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Model config
    config = EchoStreamConfig()
    
    # Build model
    model = build_echostream_model(config)
    
    # Load checkpoint
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        logger.info("Checkpoint loaded successfully")
    else:
        logger.warning("No checkpoint provided, using random initialization")
    
    model = model.to(device)
    model.eval()
    
    # Evaluator
    evaluator = EchoStreamEvaluator(model, device)
    
    # Evaluate on test set
    logger.info("\nEvaluating...")
    
    # Dummy test data (replace with actual test set)
    num_test_samples = 10
    
    for i in range(num_test_samples):
        # Generate dummy audio
        audio = torch.randn(100, 80)  # [T, F]
        
        # Evaluate
        output = evaluator.evaluate_sample(audio)
        
        evaluator.total_samples += 1
        evaluator.outputs.append(output)
        
        logger.info(f"Sample {i+1}/{num_test_samples} processed")
    
    # Compute metrics
    metrics = evaluator.compute_metrics()
    
    logger.info("\n" + "="*70)
    logger.info("Evaluation Results")
    logger.info("="*70)
    logger.info(f"Samples: {metrics['num_samples']}")
    logger.info(f"Avg Latency: {metrics['avg_latency_ms']:.2f} ms")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"\nResults saved to: {output_path}")
    
    return metrics


def evaluate_with_simuleval(args):
    """
    Evaluate using SimulEval for latency metrics.
    
    This requires SimulEval to be installed.
    """
    logger.info("="*70)
    logger.info("SimulEval Evaluation")
    logger.info("="*70)
    
    # Check if SimulEval is available
    try:
        import simuleval
    except ImportError:
        logger.error("SimulEval not installed. Install with: pip install simuleval")
        return
    
    # SimulEval command
    cmd = [
        "simuleval",
        "--agent", "agent/echostream_agent.py",
        "--source", args.source,
        "--target", args.target,
        "--model-path", args.checkpoint,
        "--output", args.output,
        "--chunk-size", str(args.chunk_size),
        "--quality-metrics", "BLEU",
        "--latency-metrics", "AL", "AP", "DAL", "StartOffset", "EndOffset", "LAAL", "ATD", "NumChunks", "RTF",
        "--device", "gpu" if torch.cuda.is_available() else "cpu",
        "--computation-aware",
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    logger.info("\nSimulEval Output:")
    logger.info(result.stdout)
    
    if result.returncode != 0:
        logger.error(f"SimulEval failed: {result.stderr}")
    else:
        logger.info("SimulEval evaluation completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoStream Evaluation")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    
    # Data
    parser.add_argument("--source", type=str, default=None, help="Source audio list")
    parser.add_argument("--target", type=str, default=None, help="Target text")
    
    # Output
    parser.add_argument("--output", type=str, default="results/metrics.json")
    
    # Evaluation mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["basic", "simuleval"],
        default="basic",
        help="Evaluation mode"
    )
    
    # SimulEval options
    parser.add_argument("--chunk-size", type=int, default=320, help="Chunk size (ms)")
    
    args = parser.parse_args()
    
    if args.mode == "basic":
        evaluate_model(args)
    elif args.mode == "simuleval":
        if not args.source or not args.target:
            parser.error("--source and --target required for SimulEval mode")
        evaluate_with_simuleval(args)

