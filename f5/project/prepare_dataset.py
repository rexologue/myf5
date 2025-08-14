import os
import argparse
import json
from pathlib import Path
import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


def main(meta_info: Path, dataset_dir: Path, save_dir: Path, dataset_name: str):
    result = []
    duration_list = []

    with open(meta_info, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Processing metadata"):
            uttr, norm_text = line.strip().split("|")
            wav_path = dataset_dir / "wavs" / f"{uttr}.wav"
            try:
                duration = sf.info(wav_path).duration
            except RuntimeError:
                print(f"Warning: Could not read audio file {wav_path}")
                duration = 0.0
            result.append({"audio_path": str(wav_path), "text": norm_text, "duration": duration})
            duration_list.append(duration)

    if not save_dir.exists():
        os.makedirs(save_dir)

    print(f"\nSaving to {save_dir} ...")
    with ArrowWriter(path=str(save_dir / "raw.arrow")) as writer:
        for item in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(item)
        writer.finalize()

    with open(save_dir / "duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert audio dataset to F5-TTS structure")
    parser.add_argument("--input", type=str, required=True, help="Input directory with LJSpeech-structured dataset")
    parser.add_argument("--repo_dir", type=str, required=True, help="F5-TTS repository directory")
    parser.add_argument("--name", type=str, required=True, help="Output dataset name")

    args = parser.parse_args()
    tokenizer = "custom"

    dataset_dir = Path(args.input)
    dataset_name = f"{args.name}_{tokenizer}"
    meta_info = dataset_dir / "metadata.csv"
    save_dir = Path(args.repo_dir) / "data" / dataset_name

    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")

    main(meta_info, dataset_dir, save_dir, dataset_name)

