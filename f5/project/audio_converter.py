import argparse
import os
import csv
import wave
import contextlib
from tqdm import tqdm
from pydub import AudioSegment


def main(input_dir: str, output_dir: str) -> None:
    """
    Converts dataset (LJSpeech-like format: wavs/*.wav + metadata.csv with 'file|text')
    to 24kHz mono 16bit WAV with normalized loudness.
    """

    input_wavs_path = os.path.join(input_dir, "wavs")
    if not os.path.exists(input_wavs_path):
        raise FileNotFoundError(f"'wavs' directory not found: {input_wavs_path}")

    input_metadata_path = os.path.join(input_dir, "metadata.csv")
    if not os.path.exists(input_metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {input_metadata_path}")

    output_wavs_path = os.path.join(output_dir, "wavs")
    os.makedirs(output_wavs_path, exist_ok=True)

    meta_rows = []
    with open(input_metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        rows = list(reader)

        for i, row in enumerate(tqdm(rows, desc="Processing files")):
            if len(row) < 2:
                print(f"⚠️  Skipping malformed line: {row}")
                continue

            input_file, text = row[0].strip(), row[1].strip()
            input_path = os.path.join(input_wavs_path, input_file)

            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Audio file not found: {input_path}")

            output_filename = f"{i:06d}.wav"
            output_path = os.path.join(output_wavs_path, output_filename)

            try:
                # Load and process audio
                audio = AudioSegment.from_file(input_path)

                # Ensure audio processing requirements
                audio = audio.set_frame_rate(24000)    # 24kHz
                audio = audio.set_sample_width(2)      # 16-bit
                audio = audio.set_channels(1)          # Mono

                # Normalize to -20 dBFS
                target_dBFS = -20.0
                change_in_dB = target_dBFS - audio.dBFS
                audio = audio.apply_gain(change_in_dB)

                # Export with strict WAV format
                audio.export(
                    output_path,
                    format="wav",
                    parameters=["-ac", "1", "-ar", "24000"]
                )

                # Verify output format
                with contextlib.closing(wave.open(output_path, "r")) as wav_file:
                    rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    sampwidth = wav_file.getsampwidth()

                    if rate != 24000 or channels != 1 or sampwidth != 2:
                        print(f"⚠️  Invalid output format: {output_path}")

                # Add to metadata (LJSpeech format: ID|text)
                meta_rows.append([output_filename[:-4], text])

            except Exception as e:
                print(f"⚠️  Error processing {input_path}: {str(e)}")

    # Write metadata
    metadata_output = os.path.join(output_dir, "metadata.csv")
    with open(metadata_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE)
        writer.writerows(meta_rows)

    print(f"✔  Completed: {len(meta_rows)} files processed")
    print(f"Output directory: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert audio dataset (LJSpeech-like) to 24kHz mono 16bit WAV format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing 'wavs/' and 'metadata.csv'"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed dataset"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise NotADirectoryError(f"Input directory not found: {args.input}")

    main(args.input, args.output)
