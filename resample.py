import os
import argparse
import numpy as np
import librosa

import scipy.io.wavfile as wav


def resample_wav_files(input_dir, output_dir, target_sr=16000):
    """Resample all .wav files under input_dir to target_sr and write to output_dir."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path, file)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Load the audio file at native sampling rate
                y, sr = librosa.load(input_path, sr=None)

                # Resample the audio to the target sample rate
                y_resampled = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)

                # Clip and convert to int16 PCM
                y_resampled = np.clip(y_resampled, -1.0, 1.0)
                wav.write(output_path, target_sr, (y_resampled * 32767).astype('int16'))


def main():
    """Parse CLI args and run resampling for noise and/or RIR datasets."""
    parser = argparse.ArgumentParser(
        description="Resample noise and RIR datasets to a target sample rate (default: 16 kHz)."
    )
    parser.add_argument(
        "--noise_dir",
        type=str,
        help="Path to the source Noise directory (recursively processed).",
    )
    parser.add_argument(
        "--rir_dir",
        type=str,
        help="Path to the source RIR directory (recursively processed).",
    )
    parser.add_argument(
        "--noise_out",
        type=str,
        help="Output directory for resampled Noise (default: <noise_dir>_16k)",
    )
    parser.add_argument(
        "--rir_out",
        type=str,
        help="Output directory for resampled RIR (default: <rir_dir>_16k)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Target sample rate. Default: 16000",
    )

    args = parser.parse_args()

    if not args.noise_dir and not args.rir_dir:
        parser.error("At least one of --noise_dir or --rir_dir must be provided.")

    if args.noise_dir:
        noise_out_dir = args.noise_out or f"{args.noise_dir.rstrip(os.sep)}_16k"
        print(f"Resampling Noise: {args.noise_dir} -> {noise_out_dir} @ {args.sr} Hz")
        resample_wav_files(args.noise_dir, noise_out_dir, target_sr=args.sr)

    if args.rir_dir:
        rir_out_dir = args.rir_out or f"{args.rir_dir.rstrip(os.sep)}_16k"
        print(f"Resampling RIR: {args.rir_dir} -> {rir_out_dir} @ {args.sr} Hz")
        resample_wav_files(args.rir_dir, rir_out_dir, target_sr=args.sr)


if __name__ == "__main__":
    main()