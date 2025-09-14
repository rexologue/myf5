#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch TTS (F5-TTS, no subprocess, no threads)
- Single model/vocoder load
- Sentence chunking via rusenttokenize ONLY + glue short ones (NO intra-sentence splits)
- Sequential generation (GPU-safe)
- Strict silence between groups only (no cross-fade, no trimming)
- Per-chunk and total duration estimates
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'F5_TTS', 'src'))

import re
import json
import inspect
import importlib
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ====== USER CONFIG ======
PATH_TO_CKPT   = "/root/model_last.pt"
PATH_TO_VOCAB  = "/root/ckpt/vocab.txt"
REF_PATH       = "/root/myf5/f5/project/refs1"   # ref.wav, ref.txt
INPUT_PATH     = "/root/myf5/f5/texts"
OUTPUT_PATH    = "/root/myf5/f5/out2"

CFG_NAME       = "/root/myf5/f5/project/F5TTS_v1_Base.yaml"
MEL_SPEC_TYPE  = "vocos"   # "vocos" | "bigvgan"
DEVICE         = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Разбиение/склейка
MIN_CHARS                 = 100     # склеиваем короткие предложения до этого порога (без пробелов)
EST_CHARS_PER_SEC         = 15.0   # только для лог-оценки длительности
SILENCE_MS_BETWEEN_GROUPS = 25    # тишина ТОЛЬКО между группами (между предложениями)

# Инференс
NFE_STEP            = 64
CFG_STRENGTH        = 2.0
SWAY_SAMPLING_COEF  = -1.0
SPEED               = 1.0
FIX_DURATION        = None

# Аудио/мел
TARGET_SAMPLE_RATE  = 24000
HOP_LENGTH          = 256
N_MEL_CHANNELS      = 100
# =========================

# ---------- sentence splitters (rusenttokenize ONLY) ----------
from rusenttokenize import ru_sent_tokenize  # type: ignore

_SENT_END_RE = re.compile(r"[\.!\?…。！？]\s*$")

def split_sentences_ru(text: str) -> List[str]:
    return [t.strip() for t in ru_sent_tokenize(text) if t and t.strip()]

def _count_chars_no_spaces(s: str) -> int:
    return len(re.sub(r"\s+", "", s))

def group_by_min_chars(sentences: List[str], min_chars: int) -> List[str]:
    """
    Склеиваем ТОЛЬКО целые предложения. Границы групп — всегда на границе предложений.
    Внутри одного предложения НЕ режем никогда.
    """
    groups: List[str] = []
    buf: List[str] = []
    acc = 0
    for s in sentences:
        if not buf:
            buf = [s]
            acc = _count_chars_no_spaces(s)
        else:
            if acc >= min_chars:
                groups.append(" ".join(buf))
                buf = [s]
                acc = _count_chars_no_spaces(s)
            else:
                buf.append(s)
                acc += _count_chars_no_spaces(s)
        if acc >= min_chars:
            groups.append(" ".join(buf))
            buf = []
            acc = 0
    if buf:
        groups.append(" ".join(buf))
    return groups

# ---------- F5-TTS imports & config ----------
from f5_tts.model.utils import convert_char_to_pinyin
from f5_tts.infer.utils_infer import load_model as _load_model, load_vocoder as _load_vocoder, preprocess_ref_audio_text

def _import_from_string(dotted: str):
    if ":" in dotted:
        mod, attr = dotted.split(":", 1)
    else:
        mod, attr = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod), attr)

def _load_config_file(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            from omegaconf import OmegaConf
            return OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Can't parse config {path}: {e}")

def _normalize_arch_keys(arch: dict) -> dict:
    mapping = {
        "dim": "dim", "depth": "depth", "heads": "heads", "ff_mult": "ff_mult",
        "text_dim": "text_dim", "conv_layers": "conv_layers",
        "qk_norm": "qk_norm", "pe_attn_head": "pe_attn_head",
        "attn_backend": "attn_backend", "attn_mask_enabled": "attn_mask_enabled",
        "checkpoint_activations": "checkpoint_activations", "dropout": "dropout",
        "long_skip_connection": "long_skip_connection", "text_mask_padding": "text_mask_padding",
    }
    return {mapping[k]: v for k, v in (arch or {}).items() if k in mapping}

import inspect
def _pick_kwargs_by_signature(cls, cfg: dict) -> dict:
    sig = inspect.signature(cls.__init__)
    allowed = {p.name for p in sig.parameters.values()
               if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    out = {k: v for k, v in cfg.items() if k in allowed}
    if "dim_head" in allowed and "dim_head" not in out:
        if "dim" in cfg and "heads" in cfg and cfg["heads"]:
            out["dim_head"] = int(cfg["dim"] // cfg["heads"])
    return out

def resolve_model_cfg(cfg_name: str):
    tried = []
    p = Path(cfg_name)
    if p.exists() and p.is_file():
        obj = _load_config_file(p)
        model_node = (obj.get("model") if isinstance(obj, dict) else None) or {}
        backbone = (model_node.get("backbone") or model_node.get("name") or "").strip()
        arch = model_node.get("arch") or {}
        if not backbone:
            raise RuntimeError(f"'model.backbone' is missing in {p}")
        backbone_map = {
            "DiT":   "f5_tts.model:DiT",
            "UNetT": "f5_tts.model:UNetT",
        }
        if backbone not in backbone_map:
            raise RuntimeError(f"Unsupported backbone '{backbone}' in {p}. Expected one of {list(backbone_map)}")
        model_cls = _import_from_string(backbone_map[backbone])
        raw_cfg = _normalize_arch_keys(arch)
        model_cfg = _pick_kwargs_by_signature(model_cls, raw_cfg)
        if not model_cfg:
            raise RuntimeError(f"Empty model_cfg after filtering by signature for backbone '{backbone}'. Raw arch: {arch}")
        return model_cls, model_cfg

    # fallbacks (если когда-то появятся пресеты в коде)
    try:
        icli = importlib.import_module("f5_tts.infer.infer_cli")
        for key in ("MODELS", "models", "MODEL_ZOO", "MODEL_FACTORY"):
            if hasattr(icli, key):
                zoo = getattr(icli, key)
                if isinstance(zoo, dict) and cfg_name in zoo:
                    return zoo[cfg_name]
        for key in ("get_model", "get_model_cfg", "build_model_and_cfg"):
            if hasattr(icli, key):
                res = getattr(icli, key)(cfg_name)
                if isinstance(res, tuple) and len(res) == 2:
                    return res
        tried.append("f5_tts.infer.infer_cli")
    except Exception as e:
        tried.append(f"f5_tts.infer.infer_cli ! ({e})")

    for modname in ("f5_tts.infer.model_configs", "f5_tts.infer.configs", "f5_tts.infer.models"):
        try:
            m = importlib.import_module(modname)
            for key in ("MODELS", "models", "MODEL_ZOO"):
                if hasattr(m, key):
                    zoo = getattr(m, key)
                    if isinstance(zoo, dict) and cfg_name in zoo:
                        return zoo[cfg_name]
            tried.append(modname)
        except Exception as e:
            tried.append(f"{modname} ! ({e})")

    raise RuntimeError(f"Can't resolve CFG '{cfg_name}'. Tried: {', '.join(tried)}")

# ---------- core TTS ----------
def prepare_ref(ref_path: str) -> Tuple[str, str]:
    ref_wav = os.path.join(ref_path, "ref.wav")
    ref_txt_path = os.path.join(ref_path, "ref.txt")
    if not os.path.exists(ref_wav):
        raise FileNotFoundError(f"Missing {ref_wav}")
    ref_text = open(ref_txt_path, "r", encoding="utf-8").read().strip() if os.path.exists(ref_txt_path) else ""
    ref_audio_proc, ref_text_norm = preprocess_ref_audio_text(ref_wav, ref_text)
    # пробел после финальной пунктуации
    if not _SENT_END_RE.search(ref_text_norm):
        if re.search(r"[\.!\?…。！？]$", ref_text_norm):
            ref_text_norm += " "
        else:
            ref_text_norm += ". "
    return ref_audio_proc, ref_text_norm

def load_ref_audio_tensor(ref_audio_path: str, target_sr: int, target_rms: float = 0.1, device: str = DEVICE):
    audio, sr = torchaudio.load(ref_audio_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * (target_rms / (rms + 1e-8))
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio); sr = target_sr
    return audio.to(device), sr

def process_chunk(
    model_obj, vocoder, ref_audio_tensor: torch.Tensor, ref_text: str, gen_text: str,
    mel_spec_type: str = MEL_SPEC_TYPE, nfe_step: int = NFE_STEP, cfg_strength: float = CFG_STRENGTH,
    sway_sampling_coef: float = SWAY_SAMPLING_COEF, speed: float = SPEED, fix_duration: Optional[float] = FIX_DURATION,
) -> Tuple[np.ndarray, np.ndarray]:
    if not ref_text.endswith(" "):
        ref_text = ref_text + " "

    local_speed = speed if len(gen_text) >= 10 else min(0.7, speed)
    text_list = [ref_text + gen_text]
    final_text_list = convert_char_to_pinyin(text_list)

    ref_audio_len = ref_audio_tensor.shape[-1] // HOP_LENGTH
    if fix_duration is not None:
        duration = int(fix_duration * TARGET_SAMPLE_RATE / HOP_LENGTH)
    else:
        ref_text_len = max(1, len(ref_text))
        gen_text_len = len(gen_text)
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / max(0.1, local_speed))

    with torch.inference_mode():
        generated, _ = model_obj.sample(
            cond=ref_audio_tensor,
            text=final_text_list,
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
        )
        generated = generated.to(torch.float32)       # [B, T_mel, n_mels]
        generated = generated[:, ref_audio_len:, :]   # убираем референс
        generated = generated.permute(0, 2, 1)        # [B, n_mels, T]

        if mel_spec_type == "vocos":
            wave = vocoder.decode(generated)
        else:
            wave = vocoder(generated)

        wave_np = wave.squeeze().cpu().numpy().astype(np.float32)
        mel_np  = generated[0].cpu().numpy()
        return wave_np, mel_np

def concat_with_silence(waves: List[np.ndarray], sr: int, silence_ms: int) -> np.ndarray:
    if not waves:
        return np.zeros(0, dtype=np.float32)
    if silence_ms <= 0:
        return np.concatenate(waves)
    gap = np.zeros(int(sr * silence_ms / 1000.0), dtype=np.float32)
    out = []
    for i, w in enumerate(waves):
        if i > 0:
            out.append(gap)
        out.append(w)
    return np.concatenate(out)

def run_for_text_file(
    txt_path: Path,
    model_obj,
    vocoder,
    ref_audio_tensor: torch.Tensor,
    ref_text: str,
    sr: int = TARGET_SAMPLE_RATE,
) -> Optional[Path]:
    raw = txt_path.read_text(encoding="utf-8").strip()
    if not raw:
        print(f"[SKIP] {txt_path.name}: empty text"); return None

    sentences = split_sentences_ru(raw)                 # 1) предложения
    groups    = group_by_min_chars(sentences, MIN_CHARS)  # 2) склейка коротышей (границы — только по предложениям)

    # Оценка длительности (грубая)
    est_secs  = [max(1, _count_chars_no_spaces(g)) / EST_CHARS_PER_SEC for g in groups]
    est_total = sum(est_secs) + (len(groups)-1) * (SILENCE_MS_BETWEEN_GROUPS/1000.0 if len(groups) > 1 else 0)
    print(f"[INFO] {txt_path.name}: sents={len(sentences)}, groups={len(groups)}; est ~{est_total:.1f}s "
          f"({', '.join(f'{t:.1f}' for t in est_secs)})")

    waves: List[np.ndarray] = []
    mels:  List[np.ndarray] = []
    for i, grp in enumerate(groups, 1):
        try:
            w, m = process_chunk(
                model_obj, vocoder, ref_audio_tensor, ref_text, grp,
                mel_spec_type=MEL_SPEC_TYPE, nfe_step=NFE_STEP, cfg_strength=CFG_STRENGTH,
                sway_sampling_coef=SWAY_SAMPLING_COEF, speed=SPEED, fix_duration=FIX_DURATION
            )
            waves.append(w); mels.append(m)
            print(f"[OK] {txt_path.name} [{i}/{len(groups)}] "
                  f"({_count_chars_no_spaces(grp)} ch, ~{len(grp)/EST_CHARS_PER_SEC:.1f}s)")
        except Exception as e:
            print(f"[ERROR] {txt_path.name} [{i}/{len(groups)}]: {e}")

    if not waves:
        print(f"[ERR] {txt_path.name}: nothing generated"); return None

    final_wave = concat_with_silence(waves, sr, SILENCE_MS_BETWEEN_GROUPS)

    out_path = Path(OUTPUT_PATH) / f"{txt_path.stem}.wav"
    sf.write(str(out_path), final_wave, sr)
    print(f"[DONE] {txt_path.name} -> {out_path} (est ~{est_total:.1f}s)")
    return out_path

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 1) Resolve model cfg & load model/vocoder
    model_cls, model_cfg = resolve_model_cfg(CFG_NAME)
    model = _load_model(
        model_cls=model_cls,
        model_cfg=model_cfg,
        ckpt_path=PATH_TO_CKPT,
        mel_spec_type=MEL_SPEC_TYPE,
        vocab_file=PATH_TO_VOCAB,
        device=DEVICE,
    )
    vocoder = _load_vocoder(vocoder_name=MEL_SPEC_TYPE, device=DEVICE)

    # 2) Prepare reference
    ref_audio_proc_path, ref_text = prepare_ref(REF_PATH)
    ref_audio_tensor, sr = load_ref_audio_tensor(ref_audio_proc_path, TARGET_SAMPLE_RATE, device=DEVICE)

    # 3) Process all texts
    input_dir = Path(INPUT_PATH)
    files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    if not files:
        print(f"[WARN] No .txt files in {INPUT_PATH}")
        return

    for p in files:
        run_for_text_file(p, model, vocoder, ref_audio_tensor, ref_text, sr=sr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
