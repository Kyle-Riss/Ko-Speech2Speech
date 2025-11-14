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
from typing import Optional
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'models'))

from echostream_model import build_echostream_model, EchoStreamConfig
from datasets import S2STManifestDataset

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
    
    # Load config
    config_dict = {}
    if args.config:
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
    
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
    
    # Dataset
    data_cfg = config_dict.get('data', {})
    config_dir = Path(args.config).resolve().parent if args.config else ROOT_DIR

    def resolve_optional_path(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = (config_dir / path).resolve()
        return str(path)

    test_manifest = args.test_manifest or data_cfg.get('test_manifest')
    test_manifest_path = resolve_optional_path(test_manifest)
    if test_manifest_path is None:
        raise ValueError("Test manifest must be provided via --test-manifest or config[data.test_manifest].")

    data_root = resolve_optional_path(data_cfg.get('data_root'))
    global_cmvn = resolve_optional_path(data_cfg.get('global_cmvn_stats_npz'))
    units_root = resolve_optional_path(data_cfg.get('units_root'))
    src_vocab_path = resolve_optional_path(data_cfg.get('src_dict'))
    tgt_vocab_path = resolve_optional_path(data_cfg.get('tgt_dict'))

    streaming_cfg = config_dict.get('streaming', {})
    streaming_chunk_ms = streaming_cfg.get('chunk_size_ms', streaming_cfg.get('chunk_size'))
    streaming_hop_ms = streaming_cfg.get('chunk_hop_ms', streaming_cfg.get('chunk_hop'))

    test_dataset = S2STManifestDataset(
        manifest_path=test_manifest_path,
        data_root=data_root,
        units_root=units_root,
        sample_rate=data_cfg.get('sample_rate', 16000),
        num_mel_bins=data_cfg.get('num_mel_bins', 80),
        src_vocab_path=src_vocab_path,
        tgt_vocab_path=tgt_vocab_path,
        text_level=data_cfg.get('tokenize_level', 'word'),
        global_cmvn_stats=global_cmvn,
        load_waveform=data_cfg.get('load_waveform', False),
        load_tgt_audio=data_cfg.get('load_tgt_audio', False),
        load_tgt_units=data_cfg.get('load_tgt_units', False),
        streaming_chunk_ms=streaming_chunk_ms,
        streaming_hop_ms=streaming_hop_ms,
        min_duration=data_cfg.get('min_duration'),
        max_duration=data_cfg.get('max_duration'),
        pad_value=data_cfg.get('pad_value', 0.0),
    )

    logger.info(f"Loaded test manifest: {test_manifest_path}")
    logger.info(f"Total samples: {len(test_dataset)}")

    # Evaluator
    evaluator = EchoStreamEvaluator(model, device)
    
    # Evaluate on test set
    logger.info("\nEvaluating...")
    
    for idx, sample in enumerate(test_dataset, start=1):
        audio = sample['speech']
        reference_text = sample.get('tgt_text')
        
        output = evaluator.evaluate_sample(audio, reference_text)
        
        evaluator.total_samples += 1
        evaluator.outputs.append(
            {
                'id': sample['id'],
                'reference_text': reference_text,
                'predictions': output,
            }
        )
        
        if idx % 10 == 0 or idx == len(test_dataset):
            logger.info(f"Processed {idx}/{len(test_dataset)} samples")
    
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

        results_payload = {
            "metrics": metrics,
            "samples": evaluator.outputs,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_payload, f, indent=2, ensure_ascii=False)

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
    
    # Configuration
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    
    # Data
    parser.add_argument("--test-manifest", type=str, default=None, help="Test manifest TSV path")
    parser.add_argument("--source", type=str, default=None, help="Source audio list (SimulEval)")
    parser.add_argument("--target", type=str, default=None, help="Target text (SimulEval)")
    
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

