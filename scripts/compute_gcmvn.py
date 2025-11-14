#!/usr/bin/env python3
"""
Compute global CMVN stats (mean/std) from a StreamSpeech/EchoStream-style TSV.

Example:
    python scripts/compute_gcmvn.py \
        --manifest data/train.tsv \
        --audio-root . \
        --output data/gcmvn.npz
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np


def load_wave(path: Path, target_sr: int = 16000):
    waveform, sample_rate = librosa.load(path.as_posix(), sr=target_sr, mono=True)
    return waveform, sample_rate


def extract_fbank(waveform, sample_rate: int, n_mels: int = 80):
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=400,
        hop_length=160,
        win_length=400,
        window="hann",
        center=True,
        power=2.0,
        n_mels=n_mels,
    )
    log_mel = librosa.power_to_db(mel + 1e-10).T  # time x dim
    return log_mel


def compute_stats(manifest: Path, audio_root: Path, output: Path, limit: int | None):
    sum_feat = None
    sum_square = None
    total_frames = 0

    with manifest.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            wav_path = audio_root / row["src_audio"]
            waveform, sample_rate = load_wave(wav_path)
            feats_np = extract_fbank(waveform, sample_rate)

            if sum_feat is None:
                feat_dim = feats_np.shape[1]
                sum_feat = np.zeros(feat_dim, dtype=np.float64)
                sum_square = np.zeros(feat_dim, dtype=np.float64)

            sum_feat += feats_np.sum(axis=0)
            sum_square += (feats_np**2).sum(axis=0)
            total_frames += feats_np.shape[0]

            if limit is not None and idx >= limit:
                break

    if total_frames == 0:
        raise RuntimeError("No frames processed. Check manifest paths.")

    mean = sum_feat / total_frames
    var = sum_square / total_frames - mean**2
    std = np.sqrt(np.maximum(var, 1e-10))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"Saved CMVN stats to {output} (frames={total_frames})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of utterances to speed up debugging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    compute_stats(args.manifest, args.audio_root, args.output, args.limit)


if __name__ == "__main__":
    main()

