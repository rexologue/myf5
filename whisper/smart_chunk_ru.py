#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import csv
import re
import unicodedata
from pathlib import Path
from typing import List, Dict

import numpy as np
import soundfile as sf
from tqdm import tqdm

import whisperx
from transformers import pipeline, AutoTokenizer
from ruaccent import RUAccent
from num2words import num2words

VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
ACUTE = "\u0301"

# ---------------------------
# ASR + word-level alignment
# ---------------------------
def load_words_whisperx(audio_path, model_name, lang, batch_size, compute_type, device="cuda"):
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    asr = model.transcribe(audio, batch_size=batch_size, language=lang)

    model_a, metadata = whisperx.load_align_model(language_code=asr["language"], device=device)
    aligned = whisperx.align(asr["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    words = []
    for seg in aligned["segments"]:
        for w in seg.get("words", []):
            if w.get("start") is not None and w.get("end") is not None and w.get("word"):
                s, e = float(w["start"]), float(w["end"])
                if e > s:
                    words.append({"start": s, "end": e, "text": w["word"]})
    words.sort(key=lambda x: x["start"])
    return words

# ---------------------------
# RUPunct (batched) — без kwargs в __call__
# ---------------------------
def load_rupunct(model_id="RUPunct/RUPunct_big", device=0, max_length=384):
    tk = AutoTokenizer.from_pretrained(model_id, strip_accents=False, add_prefix_space=True, use_fast=True)
    tk.model_max_length = max_length  # пайплайн сам применит трэнкэйт
    clf = pipeline("ner", model=model_id, tokenizer=tk, aggregation_strategy="first", device=device)
    return clf

def _rupunct_reconstruct(preds_for_one):
    def process_token(token, label):
        t = token.strip()
        m = {
            "LOWER_O": t, "LOWER_PERIOD": t+".", "LOWER_COMMA": t+",", "LOWER_QUESTION": t+"?",
            "LOWER_TIRE": t+" —", "LOWER_DVOETOCHIE": t+":", "LOWER_VOSKL": t+"!",
            "LOWER_PERIODCOMMA": t+";", "LOWER_DEFIS": t+"-", "LOWER_MNOGOTOCHIE": t+"...",
            "LOWER_QUESTIONVOSKL": t+"?!",
            "UPPER_O": t.capitalize(), "UPPER_PERIOD": t.capitalize()+".", "UPPER_COMMA": t.capitalize()+",",
            "UPPER_QUESTION": t.capitalize()+"?", "UPPER_TIRE": t.capitalize()+" —",
            "UPPER_DVOETOCHIE": t.capitalize()+":", "UPPER_VOSKL": t.capitalize()+"!",
            "UPPER_PERIODCOMMA": t.capitalize()+";", "UPPER_DEFIS": t.capitalize()+"-",
            "UPPER_MNOGOTOCHIE": t.capitalize()+"...", "UPPER_QUESTIONVOSKL": t.capitalize()+"?!",
            "UPPER_TOTAL_O": t.upper(), "UPPER_TOTAL_PERIOD": t.upper()+".", "UPPER_TOTAL_COMMA": t.upper()+",",
            "UPPER_TOTAL_QUESTION": t.upper()+"?", "UPPER_TOTAL_TIRE": t.upper()+" —",
            "UPPER_TOTAL_DVOETOCHIE": t.upper()+":", "UPPER_TOTAL_VOSKL": t.upper()+"!",
            "UPPER_TOTAL_PERIODCOMMA": t.upper()+";", "UPPER_TOTAL_DEFIS": t.upper()+"-",
            "UPPER_TOTAL_MNOGOTOCHIE": t.upper()+"...", "UPPER_TOTAL_QUESTIONVOSKL": t.upper()+"?!",
        }
        return m.get(label, t)

    out = [process_token(it["word"], it["entity_group"]) for it in preds_for_one]
    s = " ".join(out)
    s = re.sub(r"\s+([,.:;!?…])", r"\1", s)
    s = re.sub(r"\s+—", " —", s)
    return s

def rupunct_texts_batch(clf, texts: List[str], batch_size=64) -> List[str]:
    outputs = clf(texts, batch_size=batch_size)  # list[list[dict]]
    return [_rupunct_reconstruct(items) for items in outputs]

# ---------------------------
# Punctuation normalization (dedup / cleanup)
# ---------------------------
ELLIPSIS_RE = re.compile(r"…")
LONG_DOTS_RE = re.compile(r"\.{4,}")  # > '...'
SP_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
ENSURE_SPACE_AFTER_MINOR = re.compile(r"([,;:])([^\s])")
ENSURE_SPACE_AFTER_SENT = re.compile(r"([.!?])([^\s])")
BAD_PAIRS_MAP = {",.": ".", ".,": ".", ";.": ".", ".;": ".", ",;": ",", ";,": ";"}

def fix_punctuation(s: str) -> str:
    if not isinstance(s, str): return s
    t = s
    t = ELLIPSIS_RE.sub("...", t)
    t = LONG_DOTS_RE.sub("...", t)
    t = re.sub(r"([,;:!?])\1+", r"\1", t)            # !! -> !
    def collapse_periods(m):
        seq = m.group(0)
        return "..." if len(seq) >= 3 else "."
    t = re.sub(r"\.{2,}", collapse_periods, t)       # .. -> .  .... -> ...
    for bad, good in BAD_PAIRS_MAP.items():
        t = t.replace(bad, good)
    t = SP_BEFORE_PUNCT.sub(r"\1", t)
    t = ENSURE_SPACE_AFTER_MINOR.sub(r"\1 \2", t)
    t = ENSURE_SPACE_AFTER_SENT.sub(r"\1 \2", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

# ---------------------------
# Accent helpers (RUAccent)
# ---------------------------
def accents_to_plus(s: str) -> str:
    if not isinstance(s, str): return s
    if "+" in s:
        n = unicodedata.normalize("NFD", s)
        n = "".join(ch for ch in n if not unicodedata.combining(ch))
        return unicodedata.normalize("NFC", n)
    n = unicodedata.normalize("NFD", s)
    out = []
    i = 0
    while i < len(n):
        ch = n[i]
        if ch in VOWELS and i + 1 < len(n) and n[i + 1] == ACUTE:
            out.append("+"); out.append(ch); i += 2
            while i < len(n) and unicodedata.combining(n[i]):
                i += 1
        else:
            if not unicodedata.combining(ch):
                out.append(ch)
            i += 1
    return unicodedata.normalize("NFC", "".join(out))

def remove_plus(s: str) -> str:
    if not isinstance(s, str): return s
    n = unicodedata.normalize("NFD", s.replace("+", ""))
    n = "".join(ch for ch in n if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", n)

# ---------------------------
# Phrase grouping
# ---------------------------
def group_into_phrases(words: List[Dict], gap_strong=0.5):
    phrases, cur = [], []
    for i, w in enumerate(words):
        if not cur:
            cur = [w]; continue
        prev = words[i-1]
        gap = w["start"] - prev["end"]
        cur.append(w)
        if gap >= gap_strong:
            phrases.append(cur); cur = []
    if cur: phrases.append(cur)
    return phrases

# ---------------------------
# RUPunct over phrases (batched) -> eos flags
# ---------------------------
def add_rupunct_to_phrases_batch(phrases: List[List[Dict]], clf, batch_size=64):
    texts = [" ".join(w["text"] for w in ph) for ph in phrases]  # тут уже с whisper-пунктуацией
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="RUPunct (batched)"):
        chunk = texts[i:i+batch_size]
        preds = clf(chunk, batch_size=len(chunk))
        puncted = [_rupunct_reconstruct(p) for p in preds]
        for ph, txt in zip(phrases[i:i+batch_size], puncted):
            eos = bool(re.search(r"[\.!\?…]\s*$", fix_punctuation(txt)))
            out.append({"words": ph, "eos": eos})
    return out

# ---------------------------
# Pack phrases into 5-12 s chunks
# ---------------------------
def regroup_by_gaps(words, gap_strong=0.5):
    out, cur = [], []
    for i, w in enumerate(words):
        if not cur:
            cur = [w]; continue
        gap = w["start"] - words[i-1]["end"]
        cur.append(w)
        if gap >= gap_strong:
            out.append(cur); cur = []
    if cur: out.append(cur)
    return out

def pack_phrases(phrases_info, min_dur=5.0, max_dur=12.0):
    chunks = []
    i = 0
    target = min(max(10.0, min_dur), max_dur)
    def span_words(wl): return wl[0]["start"], wl[-1]["end"]

    while i < len(phrases_info):
        cur_words = phrases_info[i]["words"].copy()
        cur_eos_last = phrases_info[i]["eos"]
        i += 1

        while i < len(phrases_info):
            s0, e0 = span_words(cur_words)
            dur0 = e0 - s0
            if dur0 < min_dur:
                cur_words.extend(phrases_info[i]["words"])
                cur_eos_last = phrases_info[i]["eos"]
                i += 1
                continue
            if dur0 >= min_dur and cur_eos_last:
                break
            e1 = phrases_info[i]["words"][-1]["end"]
            if (e1 - s0) <= max_dur:
                cur_words.extend(phrases_info[i]["words"])
                cur_eos_last = phrases_info[i]["eos"]
                i += 1
            else:
                break

        s, e = span_words(cur_words)
        if (e - s) > max_dur:
            if len(cur_words) < 2:
                chunks.append(cur_words); continue
            win_left = s + 0.8 * min_dur
            win_right = s + 1.05 * max_dur
            best_k, best_score = None, -1.0
            for k in range(len(cur_words) - 1):
                g = cur_words[k+1]["start"] - cur_words[k]["end"]
                cut_t = cur_words[k]["end"]
                if win_left <= cut_t <= win_right and g > best_score:
                    best_score, best_k = g, k
            if best_k is None:
                target_t = s + target
                best_k = int(np.argmin([abs(cur_words[k]["end"] - target_t) for k in range(len(cur_words) - 1)]))
            left_words = cur_words[:best_k+1]
            right_words = cur_words[best_k+1:]
            chunks.append(left_words)
            rest_phrases = regroup_by_gaps(right_words)
            rest_infos = [{"words": ph, "eos": False} for ph in rest_phrases]
            phrases_info = rest_infos + phrases_info[i:]
            i = 0
        else:
            chunks.append(cur_words)
    return chunks

# ---------------------------
# Boundary refinement + IO
# ---------------------------
def refine_boundary(audio_path, t, win=0.25):
    with sf.SoundFile(audio_path, 'r') as f:
        sr = f.samplerate
        a = max(0.0, t - win); b = t + win
        i0 = int(a*sr); n = int((b-a)*sr)
        f.seek(i0)
        y = f.read(n, dtype='float32', always_2d=False)
    if len(y) == 0:
        return t
    frame = max(128, int(0.01*sr))
    kernel = np.ones(frame, dtype=np.float32) / max(1, frame)
    rms = np.sqrt(np.convolve(y**2, kernel, mode='same') + 1e-9)
    k_min = int(np.argmin(rms))
    left = max(1, k_min - frame)
    right = min(len(y)-1, k_min + frame)
    zc = None
    for k in range(left, right):
        if (y[k-1] <= 0.0 and y[k] > 0.0) or (y[k-1] >= 0.0 and y[k] < 0.0):
            zc = k; break
    k = zc if zc is not None else k_min
    return a + k / sr

def write_chunk(audio_path, out_path, t0, t1, fade_ms=15):
    with sf.SoundFile(audio_path, 'r') as f:
        sr = f.samplerate
        i0 = int(t0*sr); i1 = int(t1*sr)
        f.seek(i0)
        y = f.read(max(0, i1 - i0), dtype='float32', always_2d=False)
    if len(y) == 0:
        return
    fade = max(1, int(fade_ms/1000*sr))
    if len(y) >= 2*fade:
        y[:fade] *= np.linspace(0,1,fade,endpoint=False)
        y[-fade:] *= np.linspace(1,0,fade,endpoint=False)
    sf.write(out_path, y, sr, subtype='PCM_16')

# ---------------------------
# Text normalization (numbers/units/dates/time) — авто, если встретились
# ---------------------------
UNITS_MAP = {
    "кг": "килограммов", "г": "граммов", "км": "километров", "м": "метров",
    "см": "сантиметров", "мм": "миллиметров", "л": "литров", "мл": "миллилитров",
    "%": "процентов", "руб": "рублей", "руб.": "рублей", "₽": "рублей", "р.": "рублей",
    "$": "долларов", "€": "евро",
    "кб": "килобайт", "мб": "мегабайт", "гб": "гигабайт", "тб": "терабайт",
    "°c": "градусов цельсия", "°f": "градусов по фаренгейту", "°": "градусов"
}
UNITS_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s?(кг|г|км|м|см|мм|л|мл|%|руб\.?|₽|р\.|\$|€|кб|мб|гб|тб|°c|°f|°)\b", re.IGNORECASE)
NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
DATE_RE = re.compile(r"\b([0-3]?\d)[.\-/]([01]?\d)[.\-/]([12]\d{3})\b")

MONTHS_GEN = {
    1:"января",2:"февраля",3:"марта",4:"апреля",5:"мая",6:"июня",
    7:"июля",8:"августа",9:"сентября",10:"октября",11:"ноября",12:"декабря"
}

def number_to_words_ru(token: str) -> str:
    if "," in token or "." in token:
        left, right = re.split(r"[.,]", token, maxsplit=1)
        left_w = num2words(int(left), lang="ru")
        right_w = " ".join(num2words(int(d), lang="ru") for d in right)
        return f"{left_w} точка {right_w}"
    else:
        return num2words(int(token), lang="ru")

def normalize_numbers(text: str) -> str:
    if not NUM_RE.search(text): return text
    return NUM_RE.sub(lambda m: number_to_words_ru(m.group(0)), text)

def normalize_units(text: str) -> str:
    if not UNITS_RE.search(text): return text
    def repl(m):
        num, unit = m.group(1), m.group(2)
        unit_l = unit.lower()
        unit_l = unit_l.replace("КБ".lower(), "кб").replace("МБ".lower(),"мб")
        unit_l = unit_l.replace("ГБ".lower(),"гб").replace("ТБ".lower(),"тб")
        name = UNITS_MAP.get(unit_l, unit_l)
        return f"{number_to_words_ru(num)} {name}"
    return UNITS_RE.sub(repl, text)

def normalize_time(text: str) -> str:
    if not TIME_RE.search(text): return text
    def repl(m):
        h, mi = int(m.group(1)), int(m.group(2))
        return f"{num2words(h, lang='ru')} часов {num2words(mi, lang='ru')} минут"
    return TIME_RE.sub(repl, text)

def normalize_date(text: str) -> str:
    if not DATE_RE.search(text): return text
    def repl(m):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        month = MONTHS_GEN.get(mo, "")
        # упрощённо: день — кардинально (без падежа), год — кардинально + 'года'
        day = num2words(d, lang="ru")
        year = num2words(y, lang="ru")
        return f"{day} {month} {year} года".strip()
    return DATE_RE.sub(repl, text)

def normalize_all(text: str) -> str:
    # порядок: сначала units (перекрывает числа), затем time/date, затем “голые” числа
    t = normalize_units(text)
    t = normalize_time(t)
    t = normalize_date(t)
    t = normalize_numbers(t)
    return t

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--compute_type", default="float16")
    ap.add_argument("--min_dur", type=float, default=5.0)
    ap.add_argument("--max_dur", type=float, default=12.0)
    ap.add_argument("--rupunct_model", default="RUPunct/RUPunct_big")
    ap.add_argument("--rupunct_batch", type=int, default=64)
    ap.add_argument("--rupunct_maxlen", type=int, default=384)
    ap.add_argument("--accent_omograph", default="turbo3.1")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    wavs_dir = out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    # 1) ASR + align -> слова
    words = load_words_whisperx(args.audio, args.model, args.lang, args.batch_size, args.compute_type)
    if not words:
        raise RuntimeError("WhisperX не вернул ни одного слова. Проверь аудио/модель.")

    # 2) Фразировка по паузам
    phrases = group_into_phrases(words, gap_strong=0.5)

    # 3) RUPunct (батч) -> флаги конца предложения
    rupunct = load_rupunct(args.rupunct_model, device=0, max_length=args.rupunct_maxlen)
    phrases_info = add_rupunct_to_phrases_batch(phrases, rupunct, batch_size=args.rupunct_batch)

    # 4) Паковка в отрезки 5–12 c
    chunks = pack_phrases(phrases_info, args.min_dur, args.max_dur)

    # 5) Подготовим тексты чанков из whisper-слов (с их пунктуацией)
    chunk_raw_texts = [" ".join(w["text"] for w in ch) for ch in chunks]

    # 6) RUPunct (батч) для ТЕКСТОВ чанков (навешивает свою пунктуацию поверх whisper)
    chunk_texts_punct = rupunct_texts_batch(rupunct, chunk_raw_texts, batch_size=args.rupunct_batch)

    # 7) Чистим пунктуацию (выжимаем дубли) и нормализуем текст под TTS
    chunk_texts_clean = []
    for base, rp in zip(chunk_raw_texts, chunk_texts_punct):
        # объединяем через выбор RUPunct-версии как "ведущей", но сначала фиксируем пунктуацию
        text_p = fix_punctuation(rp)
        # если RUPunct дал пусто, fallback на whisper-пунктуацию
        if not text_p.strip():
            text_p = fix_punctuation(base)
        # теперь нормализация (авто, если встречается)
        text_n = normalize_all(text_p)
        chunk_texts_clean.append(text_n)

    # 8) RUAccent
    accentizer = RUAccent()
    try:
        accentizer.load(omograph_model_size=args.accent_omograph, use_dictionary=True, device="cuda", tiny_mode=False)
    except Exception:
        accentizer.load(omograph_model_size=args.accent_omograph, use_dictionary=True)

    # 9) Экспорт
    manifest_rows = []
    for idx, (ch, text_clean) in enumerate(tqdm(list(zip(chunks, chunk_texts_clean)), total=len(chunks), desc="Export chunks")):
        t0, t1 = ch[0]["start"], ch[-1]["end"]
        t0r = refine_boundary(args.audio, t0, win=0.25)
        t1r = refine_boundary(args.audio, t1, win=0.25)
        if t1r - t0r < 1.0:
            continue

        accented = accentizer.process_all(text_clean)
        accented_plus = accents_to_plus(accented)
        text_no_accent = remove_plus(accented_plus)

        filename = f"cut_{idx:05d}.wav"
        out_wav = wavs_dir / filename
        write_chunk(args.audio, str(out_wav), t0r, t1r)

        manifest_rows.append({
            "file": str("wavs" / filename),
            "start": round(t0r,3),
            "end": round(t1r,3),
            "duration": round(t1r - t0r, 3),
            "text": text_no_accent,
            "text_accented": accented_plus
        })

    # 10) metadata.csv
    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file","start","end","duration","text","text_accented"])
        w.writeheader()
        for r in manifest_rows:
            w.writerow(r)

if __name__ == "__main__":
    main()
