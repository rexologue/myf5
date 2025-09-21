import os
import time
import argparse
import pandas as pd
from typing import List, Tuple

from methods import load_f5tts_model, generate_one, save_wav

def main():
    parser = argparse.ArgumentParser("Batch TTS + RTF for LJSpeech")
    parser.add_argument("-i", "--input", required=True, help="Путь к LJSpeech датасету (metadata.csv, wavs/)")
    parser.add_argument("-o", "--output", required=True, help="Путь для вывода (будет создан metadata.csv и папка wavs)")
    parser.add_argument("--model", required=True, help="Путь к чекпоинту модели, словарю и конфигу")
    parser.add_argument("--mel-spec-type", default="vocos", choices=["vocos", "bigvgan"])
    parser.add_argument("--device", default=None, help="cuda|cpu|mps (по умолчанию авто)")
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--ref", required=True, help="Путь к референсу")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число примеров")
    args = parser.parse_args()

    in_root = args.input
    in_meta = os.path.join(in_root, "metadata.csv")
    if not os.path.exists(in_meta):
        raise FileNotFoundError(f"Не найден metadata.csv: {in_meta}")

    files = os.listdir(args.model)
    ckpt_file = [f for f in files if f.endswith(".pt")][0]
    cfg_file = [f for f in files if f.endswith(".yaml")][0]
    vocab_file = [f for f in files if f.startswith("vocab")][0]

    # Загружаем модель один раз
    MODEL = load_f5tts_model(
        ckpt_path=os.path.join(args.model, ckpt_file),
        vocab_path=os.path.join(args.model, vocab_file),
        cfg=os.path.join(args.model, cfg_file),
        mel_spec_type=args.mel_spec_type,
        device=args.device,
        target_sample_rate=args.sr,
        hop_length=args.hop,
    )
    sr = MODEL["target_sr"]

    # Референсы
    ref_wav = os.path.join(args.ref, "ref.wav")
    ref_text = open(os.path.join(args.ref, "ref.txt"), "r", encoding="utf-8").read().strip()

    # Читаем LJSpeech metadata (берем первые 2 колонки: id|text)
    df = pd.read_csv(in_meta, sep="|", header=None, dtype=str, engine="python")
    if df.shape[1] < 2:
        raise ValueError("Ожидаю как минимум две колонки в metadata.csv: id|text")
    df = df.iloc[:, :2]
    df.columns = ["id", "text"]

    if args.limit is not None:
        df = df.iloc[:args.limit].copy()

    # Готовим выход
    out_root = args.output
    out_wavs = os.path.join(out_root, "wavs")
    os.makedirs(out_wavs, exist_ok=True)

    rtf_records: List[Tuple[str, float]] = []

    # Генерации
    for idx, row in df.iterrows():
        utt_id = row["id"]
        text = row["text"] if isinstance(row["text"], str) else str(row["text"])

        t0 = time.perf_counter()
        wave, mel = generate_one(
            MODEL,
            gen_text=text,
            ref_text=ref_text,
            ref_wav_path=ref_wav,
            nfe_step=64,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
            speed=1.0,
            # fix_duration=None,
        )
        t1 = time.perf_counter()

        # RTF = время генерации / длительность аудио
        dur_sec = max(1e-9, len(wave) / float(sr))
        rtf = (t1 - t0) / dur_sec
        rtf_records.append((utt_id, rtf))

        out_wav_path = os.path.join(out_wavs, f"{utt_id}.wav")
        save_wav(wave, out_wav_path, sr=sr)

        print(f"[OK] {utt_id}: {dur_sec:.2f}s audio, {t1-t0:.2f}s gen, RTF={rtf:.3f}")

    # Сохраняем metadata.csv с RTF (id|rtf + average)
    out_meta = os.path.join(out_root, "metadata.csv")
    mean_rtf = sum(v for _, v in rtf_records) / max(1, len(rtf_records))
    with open(out_meta, "w", encoding="utf-8") as f:
        for utt_id, r in rtf_records:
            f.write(f"{utt_id}|{r:.6f}\n")
        f.write(f"average|{mean_rtf:.6f}\n")

    print(f"[DONE] Saved WAVs -> {out_wavs}")
    print(f"[DONE] Saved RTF  -> {out_meta}")
    print(f"[MEAN RTF] {mean_rtf:.3f}")

if __name__ == "__main__":
    main()

