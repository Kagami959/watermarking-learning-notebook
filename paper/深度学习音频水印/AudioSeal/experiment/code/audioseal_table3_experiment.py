"""
AudioSeal Table 3 Reproduction Experiment
==========================================
Reproduces the detection robustness experiment from AudioSeal paper Table 3.

Paper: "Proactive Detection of Voice Cloning with Localized Watermarking"
arXiv:2401.17264

For each of 15 attack types, we:
1. Watermark a set of audio samples
2. Apply the same attack to both watermarked and non-watermarked audio
3. Compute detection scores for both sets
4. Report Accuracy (best threshold), TPR, FPR, and AUC

Matches the paper's methodology:
- Balanced watermarked / non-watermarked audio clips per attack
- Threshold that maximizes accuracy
- ROC AUC
"""

import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import numpy as np
import soundfile as sf
import json
import csv
import time
import warnings
import scipy.signal as signal
from pathlib import Path
from dataclasses import dataclass, asdict
from sklearn.metrics import roc_auc_score
from typing import Dict, List

warnings.filterwarnings('ignore')


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    data_root: str = r"D:\C\Python\backdoor\LJSpeech-1.1"
    output_dir: str = r"D:\C\Python\backdoor\audiosealexp\table3_results"
    sample_rate: int = 16000
    max_duration: float = 10.0   # seconds (paper uses 10s)
    num_samples: int = 100       # per class (100 WM + 100 non-WM = 200 per attack)
    watermark_bits: int = 16
    device: str = "cpu"


# ============================================================
# Attack implementations
# ============================================================

class Attacks:
    """Attack implementations matching AudioSeal paper Appendix D.2.
    Parameters set to 'stronger than training' values for generalization testing."""

    @staticmethod
    def none(audio: np.ndarray, sr: int) -> np.ndarray:
        return audio.copy()

    @staticmethod
    def bandpass(audio: np.ndarray, sr: int) -> np.ndarray:
        nyq = sr / 2
        b, a = signal.butter(4, [300 / nyq, min(3400 / nyq, 0.99)], btype='band')
        return signal.filtfilt(b, a, audio).astype(np.float32)

    @staticmethod
    def highpass(audio: np.ndarray, sr: int) -> np.ndarray:
        nyq = sr / 2
        b, a = signal.butter(4, min(3000 / nyq, 0.99), btype='high')
        return signal.filtfilt(b, a, audio).astype(np.float32)

    @staticmethod
    def lowpass(audio: np.ndarray, sr: int) -> np.ndarray:
        nyq = sr / 2
        b, a = signal.butter(4, min(1000 / nyq, 0.99), btype='low')
        return signal.filtfilt(b, a, audio).astype(np.float32)

    @staticmethod
    def boost(audio: np.ndarray, sr: int) -> np.ndarray:
        return np.clip(audio * 2.0, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def duck(audio: np.ndarray, sr: int) -> np.ndarray:
        return (audio * 0.2).astype(np.float32)

    @staticmethod
    def echo(audio: np.ndarray, sr: int) -> np.ndarray:
        delay_samples = int(0.1 * sr)
        output = audio.copy()
        if len(audio) > delay_samples:
            output[delay_samples:] += 0.5 * audio[:-delay_samples]
        return output.astype(np.float32)

    @staticmethod
    def pink_noise(audio: np.ndarray, sr: int) -> np.ndarray:
        n = len(audio)
        white = np.random.randn(n).astype(np.float32)
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        freqs[0] = 1
        S = np.fft.rfft(white)
        S /= np.sqrt(freqs)
        pink = np.fft.irfft(S, n=n).astype(np.float32)
        sig_power = np.mean(audio ** 2)
        noise_power = sig_power / (10 ** (20.0 / 10))
        pink = pink * np.sqrt(noise_power / (np.mean(pink ** 2) + 1e-10))
        return (audio + pink).astype(np.float32)

    @staticmethod
    def white_noise(audio: np.ndarray, sr: int) -> np.ndarray:
        sig_power = np.mean(audio ** 2)
        noise_power = sig_power / (10 ** (20.0 / 10))
        noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
        return (audio + noise).astype(np.float32)

    @staticmethod
    def fast_1_25x(audio: np.ndarray, sr: int) -> np.ndarray:
        n_target = int(len(audio) / 1.25)
        resampled = signal.resample(audio, n_target)
        output = np.zeros_like(audio)
        copy_len = min(len(resampled), len(audio))
        output[:copy_len] = resampled[:copy_len]
        return output.astype(np.float32)

    @staticmethod
    def smooth(audio: np.ndarray, sr: int) -> np.ndarray:
        kernel = np.ones(5) / 5.0
        return np.convolve(audio, kernel, mode='same').astype(np.float32)

    @staticmethod
    def resample(audio: np.ndarray, sr: int) -> np.ndarray:
        n_down = len(audio) // 2
        downsampled = signal.resample(audio, n_down)
        upsampled = signal.resample(downsampled, len(audio))
        return upsampled.astype(np.float32)

    @staticmethod
    def aac(audio: np.ndarray, sr: int) -> np.ndarray:
        nyq = sr / 2
        b, a = signal.butter(4, [100 / nyq, min(6000 / nyq, 0.99)], btype='band')
        filtered = signal.filtfilt(b, a, audio)
        sig_power = np.mean(filtered ** 2)
        noise_power = sig_power / (10 ** (30.0 / 10))
        noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
        return (filtered + noise).astype(np.float32)

    @staticmethod
    def mp3(audio: np.ndarray, sr: int) -> np.ndarray:
        nyq = sr / 2
        b, a = signal.butter(6, min(4000 / nyq, 0.99), btype='low')
        filtered = signal.filtfilt(b, a, audio)
        sig_power = np.mean(filtered ** 2)
        noise_power = sig_power / (10 ** (25.0 / 10))
        noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
        return (filtered + noise).astype(np.float32)

    @staticmethod
    def encodec(audio: np.ndarray, sr: int) -> np.ndarray:
        sig_power = np.mean(audio ** 2)
        noise_power = sig_power / (10 ** (35.0 / 10))
        noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
        return (audio + noise).astype(np.float32)


ATTACK_REGISTRY = {
    "none": Attacks.none,
    "bandpass": Attacks.bandpass,
    "highpass": Attacks.highpass,
    "lowpass": Attacks.lowpass,
    "boost": Attacks.boost,
    "duck": Attacks.duck,
    "echo": Attacks.echo,
    "pink": Attacks.pink_noise,
    "white": Attacks.white_noise,
    "fast_1.25x": Attacks.fast_1_25x,
    "smooth": Attacks.smooth,
    "resample": Attacks.resample,
    "aac": Attacks.aac,
    "mp3": Attacks.mp3,
    "encodec": Attacks.encodec,
}


# ============================================================
# Audio quality metrics
# ============================================================

def compute_si_snr(reference: np.ndarray, degraded: np.ndarray) -> float:
    ref = reference - np.mean(reference)
    deg = degraded - np.mean(degraded)
    dot = np.dot(ref, deg)
    norm_sq = np.dot(ref, ref) + 1e-10
    proj = (dot / norm_sq) * ref
    noise = deg - proj
    si_snr = 10 * np.log10(np.sum(proj ** 2) / (np.sum(noise ** 2) + 1e-10))
    return float(si_snr)


def compute_pesq_score(reference: np.ndarray, degraded: np.ndarray, sr: int = 16000) -> float:
    try:
        from pesq import pesq as pesq_func
        return float(pesq_func(sr, reference, degraded, 'wb'))
    except Exception:
        return -1.0


def compute_stoi_score(reference: np.ndarray, degraded: np.ndarray, sr: int = 16000) -> float:
    try:
        from pystoi import stoi as stoi_func
        min_len = min(len(reference), len(degraded))
        return float(stoi_func(reference[:min_len], degraded[:min_len], sr, extended=False))
    except Exception:
        return -1.0


# ============================================================
# Main experiment
# ============================================================

class Table3Experiment:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        os.makedirs(config.output_dir, exist_ok=True)

        print("Loading AudioSeal models...")
        from audioseal import AudioSeal
        self.generator = AudioSeal.load_generator('audioseal_wm_16bits')
        self.detector = AudioSeal.load_detector('audioseal_detector_16bits')
        self.generator.to(self.device).eval()
        self.detector.to(self.device).eval()
        print("Models loaded successfully.")

        wav_dir = Path(config.data_root) / "wavs"
        all_wavs = sorted(list(wav_dir.glob("*.wav")))
        self.audio_files = all_wavs[:config.num_samples * 2]
        print(f"Found {len(all_wavs)} WAV files, using {len(self.audio_files)}")

    def load_audio(self, filepath: str) -> np.ndarray:
        audio, orig_sr = sf.read(filepath, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if orig_sr != self.config.sample_rate:
            n_target = int(len(audio) * self.config.sample_rate / orig_sr)
            audio = signal.resample(audio, n_target)
        max_samples = int(self.config.max_duration * self.config.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        return audio

    def watermark_audio(self, audio: np.ndarray) -> np.ndarray:
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
        msg = torch.randint(0, 2, (1, self.config.watermark_bits)).to(self.device)
        with torch.no_grad():
            watermarked = self.generator(audio_tensor, sample_rate=self.config.sample_rate, message=msg)
        return watermarked.squeeze().cpu().numpy()

    def detect_watermark(self, audio: np.ndarray) -> float:
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            result, msg = self.detector(audio_tensor, sample_rate=self.config.sample_rate)
        # Channel 1 = P(watermark)
        return float(result[0, 1, :].mean().item())

    def compute_metrics(self, all_scores: np.ndarray, all_labels: np.ndarray):
        """Compute accuracy (best threshold), TPR, FPR, AUC"""
        # AUC
        try:
            auc = roc_auc_score(all_labels, all_scores)
        except Exception:
            auc = 0.0

        # Find optimal threshold via ROC curve
        best_acc = 0.0
        best_threshold = 0.5
        best_tpr = 0.0
        best_fpr = 0.0

        for threshold in np.arange(0.01, 1.00, 0.01):
            predictions = (all_scores >= threshold).astype(int)
            tp = np.sum((predictions == 1) & (all_labels == 1))
            fn = np.sum((predictions == 0) & (all_labels == 1))
            fp = np.sum((predictions == 1) & (all_labels == 0))
            tn = np.sum((predictions == 0) & (all_labels == 0))
            acc = (tp + tn) / (tp + tn + fp + fn + 1e-10)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
                best_tpr = tp / (tp + fn + 1e-10)
                best_fpr = fp / (fp + tn + 1e-10)

        return {
            "accuracy": float(best_acc),
            "tpr": float(best_tpr),
            "fpr": float(best_fpr),
            "auc": float(auc),
            "best_threshold": float(best_threshold),
        }

    def run(self):
        attack_names = list(ATTACK_REGISTRY.keys())
        n_per_class = self.config.num_samples
        results = {}

        for attack_name in attack_names:
            attack_fn = ATTACK_REGISTRY[attack_name]
            print(f"\n{'='*60}")
            print(f"Attack: {attack_name} ({n_per_class} samples per class)")
            print('='*60)

            wm_scores = []
            orig_scores = []
            quality_si_snr = []
            quality_pesq = []
            quality_stoi = []

            t0 = time.time()

            for i in range(n_per_class):
                if i % 20 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (n_per_class - i - 1) / rate if rate > 0 else 0
                    print(f"  [{i+1}/{n_per_class}] {rate:.1f} samples/s, ETA {eta:.0f}s")

                try:
                    audio = self.load_audio(str(self.audio_files[i]))

                    # Watermarked
                    wm_audio = self.watermark_audio(audio)
                    wm_attacked = attack_fn(wm_audio, self.config.sample_rate)
                    wm_score = self.detect_watermark(wm_attacked)
                    wm_scores.append(wm_score)

                    # Original (non-watermarked)
                    orig_attacked = attack_fn(audio, self.config.sample_rate)
                    orig_score = self.detect_watermark(orig_attacked)
                    orig_scores.append(orig_score)

                    # Quality: compare attacked watermarked vs original
                    min_len = min(len(audio), len(wm_attacked))
                    si_snr = compute_si_snr(audio[:min_len], wm_attacked[:min_len])
                    quality_si_snr.append(si_snr)

                except Exception as e:
                    print(f"  Error on sample {i}: {e}")
                    continue

            elapsed = time.time() - t0

            wm_scores = np.array(wm_scores)
            orig_scores = np.array(orig_scores)
            all_scores = np.concatenate([wm_scores, orig_scores])
            all_labels = np.concatenate([np.ones(len(wm_scores)), np.zeros(len(orig_scores))])

            metrics = self.compute_metrics(all_scores, all_labels)
            result = {
                **metrics,
                "wm_scores_mean": float(wm_scores.mean()),
                "wm_scores_std": float(wm_scores.std()),
                "orig_scores_mean": float(orig_scores.mean()),
                "orig_scores_std": float(orig_scores.std()),
                "n_samples": len(wm_scores),
                "time_seconds": elapsed,
                "quality": {
                    "si_snr_mean": float(np.mean(quality_si_snr)) if quality_si_snr else -1,
                    "si_snr_std": float(np.std(quality_si_snr)) if quality_si_snr else -1,
                },
            }
            results[attack_name] = result

            print(f"  Acc={metrics['accuracy']:.3f} TPR={metrics['tpr']:.3f} "
                  f"FPR={metrics['fpr']:.3f} AUC={metrics['auc']:.3f}")
            print(f"  SI-SNR={result['quality']['si_snr_mean']:.1f} dB")
            print(f"  Time: {elapsed:.1f}s")

            # Save intermediate results
            with open(os.path.join(self.config.output_dir, "intermediate_results.json"), "w") as f:
                json.dump(results, f, indent=2)

        # Compute averages
        attack_results = {k: v for k, v in results.items() if k in attack_names}
        results["average"] = {
            "accuracy": float(np.mean([r["accuracy"] for r in attack_results.values()])),
            "tpr": float(np.mean([r["tpr"] for r in attack_results.values()])),
            "fpr": float(np.mean([r["fpr"] for r in attack_results.values()])),
            "auc": float(np.mean([r["auc"] for r in attack_results.values()])),
        }

        # Save final results
        output = {"config": asdict(self.config), "results": results}
        with open(os.path.join(self.config.output_dir, "table3_results.json"), "w") as f:
            json.dump(output, f, indent=2)

        # Save CSV
        with open(os.path.join(self.config.output_dir, "table3_results.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Attack", "Accuracy", "TPR", "FPR", "AUC", "SI-SNR", "N"])
            for name in attack_names:
                r = results[name]
                q = r.get("quality", {})
                writer.writerow([
                    name,
                    f"{r['accuracy']:.4f}", f"{r['tpr']:.4f}", f"{r['fpr']:.4f}",
                    f"{r['auc']:.4f}", f"{q.get('si_snr_mean', -1):.1f}", r["n_samples"],
                ])
            avg = results["average"]
            writer.writerow(["Average",
                f"{avg['accuracy']:.4f}", f"{avg['tpr']:.4f}", f"{avg['fpr']:.4f}",
                f"{avg['auc']:.4f}", "", ""])

        # Print comparison table
        self._print_comparison(results, attack_names)

        print(f"\nResults saved to {self.config.output_dir}")
        return results

    def _print_comparison(self, results, attack_names):
        paper = {
            "none":      (1.00, 1.00, 0.00, 1.00),
            "bandpass":  (1.00, 1.00, 0.00, 1.00),
            "highpass":  (0.61, 0.82, 0.60, 0.61),
            "lowpass":   (0.99, 0.99, 0.00, 0.99),
            "boost":     (1.00, 1.00, 0.00, 1.00),
            "duck":      (1.00, 1.00, 0.00, 1.00),
            "echo":      (1.00, 1.00, 0.00, 1.00),
            "pink":      (1.00, 1.00, 0.00, 1.00),
            "white":     (0.91, 0.86, 0.04, 0.95),
            "fast_1.25x":(0.99, 0.99, 0.00, 1.00),
            "smooth":    (0.99, 0.99, 0.00, 1.00),
            "resample":  (1.00, 1.00, 0.00, 1.00),
            "aac":       (1.00, 1.00, 0.00, 1.00),
            "mp3":       (1.00, 1.00, 0.00, 1.00),
            "encodec":   (0.98, 0.98, 0.01, 0.96),
        }

        print("\n" + "=" * 90)
        print("COMPARISON: Our Results vs Paper Table 3")
        print("=" * 90)
        print(f"{'Attack':<12} {'Our Acc':>8} {'Paper':>7} {'Our AUC':>8} {'Paper':>7} "
              f"{'Our TPR':>8} {'Paper':>7} {'Our FPR':>8} {'Paper':>7}")
        print("-" * 90)

        for name in attack_names:
            r = results[name]
            p = paper.get(name, (0, 0, 0, 0))
            print(f"{name:<12} {r['accuracy']:>8.3f} {p[0]:>7.2f} "
                  f"{r['auc']:>8.3f} {p[3]:>7.2f} "
                  f"{r['tpr']:>8.3f} {p[1]:>7.2f} "
                  f"{r['fpr']:>8.3f} {p[2]:>7.2f}")

        avg = results["average"]
        print("-" * 90)
        print(f"{'Average':<12} {avg['accuracy']:>8.3f} {'0.98':>7} "
              f"{avg['auc']:>8.3f} {'0.97':>7} "
              f"{avg['tpr']:>8.3f} {'0.98':>7} "
              f"{avg['fpr']:>8.3f} {'0.04':>7}")
        print("=" * 90)


if __name__ == "__main__":
    config = Config()
    experiment = Table3Experiment(config)
    experiment.run()
