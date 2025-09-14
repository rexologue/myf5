import os
import sys
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

from formater import format_text  # оставил на случай препроцесса

# === Config ===
PATH_TO_CKPT  = "/root/ckpt/model_last.pt"
PATH_TO_VOCAB = "/root/ckpt/vocab.txt"
REF_PATH      = "/root/myf5/f5/project/refs1"
CFG           = "F5TTS_v1_Base"
DEVICE        = "cuda"             # или "cpu"
OUTPUT_PATH   = "/root/myf5/f5/out"
INPUT_PATH    = "/root/myf5/f5/texts"

SILENCE_DURATION_MS = 150          # пауза между аудио-кусочками
MIN_CHARS = 40                     # минимальная длина (символов, без пробелов) для генерации
COUNT_SPACES = False               # учитывать ли пробелы в MIN_CHARS (False = игнорировать)
ESTIMATE_MODE = "chars"            # "chars" или "syll"
EST_CHARS_PER_SEC = 15.0           # для оценки длительности по символам
EST_SYLL_PER_SEC  = 5.5            # для оценки длительности по слогам

# === Sentence splitter (ru) ===
try:
    from rusenttokenize import ru_sent_tokenize  # type: ignore
    def split_sentences(text: str) -> List[str]:
        sents = [s.strip() for s in ru_sent_tokenize(text) if s.strip()]
        return sents
except Exception:
    _SENT_RE = re.compile(r'([^.!?…]+[.!?…]+)|([^.!?…]+$)', flags=re.U)
    def split_sentences(text: str) -> List[str]:
        parts = [("".join(t)).strip() for t in _SENT_RE.findall(text)]
        return [p for p in parts if p]

# === Helpers ===
def read_ref_text(ref_dir: str) -> str:
    with open(os.path.join(ref_dir, "ref.txt"), "r", encoding="utf-8") as f:
        return f.read().strip()

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def gen_tmp_name(tmp_dir: Path, base: str, idx: int) -> Path:
    return tmp_dir / f"{base}_{idx:03d}.wav"

def concat_with_silence(wav_paths: List[Path], silence_ms: int) -> Tuple[np.ndarray, int]:
    """Читает список wav'ов, конкатенирует с тишиной между ними. Возвращает (audio, sr)."""
    if not wav_paths:
        raise ValueError("Nothing to concatenate")

    audio_list = []
    sr_ref = None
    ch_ref = None

    for p in wav_paths:
        data, sr = sf.read(p, dtype="float32", always_2d=True)  # shape: (N, C)
        if sr_ref is None:
            sr_ref = sr
            ch_ref = data.shape[1]
        else:
            if sr != sr_ref:
                raise ValueError(f"Sample rate mismatch: {p} has {sr}, expected {sr_ref}")
            if data.shape[1] != ch_ref:
                if data.shape[1] == 1 and ch_ref == 2:
                    data = np.repeat(data, 2, axis=1)
                elif data.shape[1] == 2 and ch_ref == 1:
                    data = data.mean(axis=1, keepdims=True)

        audio_list.append(data)

    silence_len = int(sr_ref * (silence_ms / 1000.0))
    silence = np.zeros((silence_len, ch_ref), dtype=np.float32)

    pieces = []
    for i, a in enumerate(audio_list):
        if i > 0:
            pieces.append(silence)
        pieces.append(a)

    out = np.vstack(pieces)
    return out, sr_ref  # type: ignore[return-value]

def run_infer_cli(sentence: str, out_file: Path, paths: dict) -> None:
    """
    Вызывает f5_tts.infer.infer_cli для одного куска текста.
    Пытаемся задать имя выходного файла через -w; делаем fallback.
    """
    command = [
        sys.executable, "-m", "f5_tts.infer.infer_cli",
        "-p", paths["ckpt"],
        "-m", paths["cfg"],
        "-v", paths["vocab"],
        "-r", paths["ref_wav"],
        "-s", paths["ref_text"],
        "-t", sentence,
        "-o", paths["out_dir"],
        "-w", str(out_file.name),
        "--device", paths["device"],
    ]
    subprocess.run(command, check=True)

def char_count(s: str) -> int:
    return len(s) if COUNT_SPACES else len(re.sub(r"\s+", "", s))

def count_syllables_ru(s: str) -> int:
    # грубо: считаем гласные как слоги
    return len(re.findall(r"[аеёиоуыэюяAEËIOUYÄÖÜaeiouy]", s, flags=re.IGNORECASE))

def estimate_duration_sec(s: str) -> float:
    if ESTIMATE_MODE == "syll":
        n = max(1, count_syllables_ru(s))
        return n / EST_SYLL_PER_SEC
    else:
        n = max(1, char_count(s))
        return n / EST_CHARS_PER_SEC

def group_by_min_chars(sents: List[str], min_chars: int) -> List[str]:
    """Склеивает короткие предложения с последующими, пока длина (по правилу char_count) не >= min_chars."""
    if min_chars <= 0:
        return sents[:]
    groups = []
    buf = ""
    for s in sents:
        if not s.strip():
            continue
        if not buf:
            buf = s.strip()
        else:
            # если текущий буфер уже достаточно длинный — зафиксируем и начнём новый
            if char_count(buf) >= min_chars:
                groups.append(buf)
                buf = s.strip()
            else:
                buf = (buf + " " + s.strip()).strip()
        # если после добавления стало достаточно — фиксируем
        if buf and char_count(buf) >= min_chars:
            groups.append(buf)
            buf = ""

    if buf:  # остаток
        # можно либо отправить как есть, либо приклеить к последнему — оставим как есть
        groups.append(buf)
    return groups

def main():
    REF_TEXT = read_ref_text(REF_PATH)
    os.chdir("F5_TTS/src")
    ensure_dir(OUTPUT_PATH)

    texts = sorted(os.listdir(INPUT_PATH))
    for text_file in texts:
        in_path = os.path.join(INPUT_PATH, text_file)
        if not os.path.isfile(in_path):
            continue

        number = Path(text_file).stem  # имя итогового файла
        raw_text = load_text(in_path)
        # raw_text = format_text(raw_text)  # при необходимости

        base_sents = split_sentences(raw_text)
        if not base_sents:
            print(f"[SKIP] {text_file}: пустой текст после токенизации")
            continue

        # 1) Склеиваем короткие в группы по MIN_CHARS
        groups = group_by_min_chars(base_sents, MIN_CHARS)

        # 2) Оценка длительности по группам (лог)
        est_group_secs = [estimate_duration_sec(g) for g in groups]
        est_total = sum(est_group_secs) + max(0, len(groups) - 1) * (SILENCE_DURATION_MS / 1000.0)
        print(f"[INFO] {text_file}: {len(base_sents)} предлож., {len(groups)} групп после MIN_CHARS={MIN_CHARS}. "
              f"Оценка длит.: ~{est_total:.1f}s "
              f"(группы: {', '.join(f'{t:.1f}' for t in est_group_secs)})")

        tmp_dir = Path(OUTPUT_PATH) / f"{number}_parts"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "ckpt": PATH_TO_CKPT,
            "cfg": CFG,
            "vocab": PATH_TO_VOCAB,
            "ref_wav": os.path.join(REF_PATH, "ref.wav"),
            "ref_text": REF_TEXT,
            "out_dir": str(tmp_dir),
            "device": DEVICE,
        }

        part_paths: List[Path] = []
        for i, chunk in enumerate(groups, start=1):
            part_out = gen_tmp_name(tmp_dir, number, i)
            try:
                run_infer_cli(chunk, part_out, paths)
                fallback = tmp_dir / "infer_cli_basic.wav"
                if not part_out.exists() and fallback.exists():
                    fallback.rename(part_out)
                if part_out.exists():
                    part_paths.append(part_out)
                    print(f"[OK] {text_file} :: [{i}/{len(groups)}] "
                          f"({char_count(chunk)} ch, ~{estimate_duration_sec(chunk):.1f}s)")
                else:
                    print(f"[WARN] {text_file} :: [{i}/{len(groups)}] — отсутствует выходной файл")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] CLI failed on chunk {i}: {e}")
            except Exception as e:
                print(f"[ERROR] Unexpected on chunk {i}: {e}")

        if not part_paths:
            print(f"[ERROR] {text_file}: нет валидных кусков — пропуск")
            continue

        # 3) Сшить в финальный wav с паузами
        try:
            audio, sr = concat_with_silence(part_paths, SILENCE_DURATION_MS)
            final_path = Path(OUTPUT_PATH) / f"{number}.wav"
            sf.write(final_path, audio, sr)
            print(f"[DONE] {text_file} → {final_path} (оценка ~{est_total:.1f}s)")
        except Exception as e:
            print(f"[ERROR] Concat failed for {text_file}: {e}")

        # shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
