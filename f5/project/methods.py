import os
import re
import json
import importlib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
import torchaudio
import numpy as np
import soundfile as sf

from f5_tts.model.utils import convert_char_to_pinyin
from f5_tts.infer.utils_infer import (
    load_model as _load_model,
    load_vocoder as _load_vocoder,
    preprocess_ref_audio_text,
)

_PUNCT_END_RE = re.compile(r"[\.!\?…。！？]\s*$")


# -------------------------- 1) ЗАГРУЗКА МОДЕЛИ -------------------------- #
def _import_from_string(dotted: str):
    if ":" in dotted:
        mod, attr = dotted.split(":", 1)
    else:
        mod, attr = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod), attr)

def _read_cfg_from_file(cfg_path: Path) -> Dict[str, Any]:
    if cfg_path.suffix.lower() == ".json":
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    try:
        import yaml
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(str(cfg_path)), resolve=True)  # type: ignore

def _normalize_arch_keys(arch: dict) -> dict:
    # Пропускаем только те ключи, что реально есть в сигнатуре моделей
    mapping = {
        "dim": "dim", "depth": "depth", "heads": "heads", "ff_mult": "ff_mult",
        "text_dim": "text_dim", "conv_layers": "conv_layers",
        "qk_norm": "qk_norm", "pe_attn_head": "pe_attn_head",
        "attn_backend": "attn_backend", "attn_mask_enabled": "attn_mask_enabled",
        "checkpoint_activations": "checkpoint_activations", "dropout": "dropout",
        "long_skip_connection": "long_skip_connection", "text_mask_padding": "text_mask_padding",
    }
    return {mapping[k]: v for k, v in (arch or {}).items() if k in mapping}

def _pick_kwargs_by_signature(cls, cfg: dict) -> dict:
    import inspect
    sig = inspect.signature(cls.__init__)
    allowed = {p.name for p in sig.parameters.values()
               if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    out = {k: v for k, v in cfg.items() if k in allowed}
    if "dim_head" in allowed and "dim_head" not in out:
        if "dim" in cfg and "heads" in cfg and cfg["heads"]:
            out["dim_head"] = int(cfg["dim"] // cfg["heads"])
    return out

def _resolve_model_cfg(cfg: str) -> Tuple[type, dict]:
    """
    Поддерживает:
      - путь к .yaml/.yml/.json с узлами model.backbone и model.arch
      - или пресетное имя из f5_tts (например, 'F5TTS_v1_Base') — попробуем найти в зоопарках.
    """
    p = Path(cfg)
    if p.exists() and p.is_file():
        obj = _read_cfg_from_file(p)
        model_node = (obj.get("model") if isinstance(obj, dict) else None) or {}
        backbone = (model_node.get("backbone") or model_node.get("name") or "").strip()
        arch = model_node.get("arch") or {}
        if not backbone:
            raise RuntimeError(f"'model.backbone' не найден в {p}")
        backbone_map = {
            "DiT":   "f5_tts.model:DiT",
            "UNetT": "f5_tts.model:UNetT",
        }
        if backbone not in backbone_map:
            raise RuntimeError(f"Неподдерживаемый backbone '{backbone}'. Ожидались: {list(backbone_map)}")
        model_cls = _import_from_string(backbone_map[backbone])
        model_cfg = _pick_kwargs_by_signature(model_cls, _normalize_arch_keys(arch))
        if not model_cfg:
            raise RuntimeError(f"Пустой model_cfg после фильтрации. Исходный arch: {arch}")
        return model_cls, model_cfg

    # Пресет из кода
    tried = []
    for modname in ("f5_tts.infer.infer_cli",
                    "f5_tts.infer.model_configs",
                    "f5_tts.infer.configs",
                    "f5_tts.infer.models"):
        try:
            m = importlib.import_module(modname)
            for key in ("MODELS", "models", "MODEL_ZOO", "MODEL_FACTORY"):
                if hasattr(m, key):
                    zoo = getattr(m, key)
                    if isinstance(zoo, dict) and cfg in zoo:
                        return zoo[cfg]
            for key in ("get_model", "get_model_cfg", "build_model_and_cfg"):
                if hasattr(m, key):
                    res = getattr(m, key)(cfg)
                    if isinstance(res, tuple) and len(res) == 2:
                        return res
            tried.append(modname)
        except Exception as e:
            tried.append(f"{modname}! ({e})")
    raise RuntimeError(f"Не удалось разрешить CFG '{cfg}'. Пробовали: {', '.join(tried)}")

def load_f5tts_model(
    ckpt_path: str,
    vocab_path: str,
    cfg: str,                           # путь к yaml/JSON или имя пресета (напр., "F5TTS_v1_Base")
    mel_spec_type: str = "vocos",       # "vocos" | "bigvgan"
    device: Optional[str] = None,
    target_sample_rate: int = 24000,
    hop_length: int = 256,
):
    """
    Возвращает словарь с уже загруженными model, vocoder и служебными полями.
    """
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else
            ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    model_cls, model_cfg = _resolve_model_cfg(cfg)
    model = _load_model(
        model_cls=model_cls,
        model_cfg=model_cfg,
        ckpt_path=ckpt_path,
        mel_spec_type=mel_spec_type,
        vocab_file=vocab_path,
        device=device,
    )
    vocoder = _load_vocoder(vocoder_name=mel_spec_type, device=device)
    return {
        "model": model,
        "vocoder": vocoder,
        "device": device,
        "target_sr": target_sample_rate,
        "hop_length": hop_length,
        "mel_spec_type": mel_spec_type,
    }


# -------------------------- 2) ОДНА ГЕНЕРАЦИЯ -------------------------- #
def _ensure_trailing_punct_space(s: str) -> str:
    # гарантируем финальную пунктуацию + пробел (для конкатенации промпта)
    if not _PUNCT_END_RE.search(s):
        if re.search(r"[\.!\?…。！？]$", s):
            s += " "
        else:
            s += ". "
    return s

def _load_ref_audio_tensor(
    ref_audio_path: str,
    target_sr: int,
    device: str,
    target_rms: float = 0.1,
) -> Tuple[torch.Tensor, int]:
    audio, sr = torchaudio.load(ref_audio_path)  # [C, T]
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)  # mono
    # лёгкая нормализация по громкости, если слишком тихо
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * (target_rms / (rms + 1e-8))
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio); sr = target_sr
    return audio.to(device), sr

def generate_one(
    model_bundle: Dict[str, Any],
    gen_text: str,
    ref_text: str,
    ref_wav_path: str,
    *,
    nfe_step: int = 64,
    cfg_strength: float = 2.0,
    sway_sampling_coef: float = -1.0,
    speed: float = 1.0,
    fix_duration: Optional[float] = None,   # в секундах, если хочешь жёстко задать
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает (wave_np, mel_np) для одного запроса.
    """
    model      = model_bundle["model"]
    vocoder    = model_bundle["vocoder"]
    device     = model_bundle["device"]
    target_sr  = model_bundle["target_sr"]
    hop_length = model_bundle["hop_length"]
    mel_spec   = model_bundle["mel_spec_type"]

    # 1) Подготовка эталона (путь до предобработанного ref.wav + нормализованный ref_text)
    ref_audio_proc_path, ref_text_norm = preprocess_ref_audio_text(ref_wav_path, ref_text or "")
    ref_text_norm = _ensure_trailing_punct_space(ref_text_norm)

    # 2) Загружаем эталон в тензор
    ref_audio_tensor, _sr = _load_ref_audio_tensor(ref_audio_proc_path, target_sr=target_sr, device=device)

    # 3) Готовим текст (префикс: ref_text_norm + генерируемый текст)
    if not ref_text_norm.endswith(" "):
        ref_text_norm += " "
    local_speed = speed if len(gen_text) >= 10 else min(0.7, speed)
    text_list = [ref_text_norm + gen_text]
    final_text_list = convert_char_to_pinyin(text_list)

    # 4) Прикидка длительности в мел-кадрах
    ref_audio_len = ref_audio_tensor.shape[-1] // hop_length
    if fix_duration is not None:
        duration = int(fix_duration * target_sr / hop_length)
    else:
        ref_len = max(1, len(ref_text_norm))
        gen_len = len(gen_text)
        duration = ref_audio_len + int(ref_audio_len / ref_len * gen_len / max(0.1, local_speed))

    # 5) Инференс
    with torch.inference_mode():
        generated, _ = model.sample(
            cond=ref_audio_tensor,
            text=final_text_list,
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
        )
        # [B, T_mel, n_mels] -> оставляем только сгенерированную часть (без референса) и приводим к [B, n_mels, T]
        generated = generated.to(torch.float32)
        generated = generated[:, ref_audio_len:, :]
        generated = generated.permute(0, 2, 1)

        if mel_spec == "vocos":
            wave = vocoder.decode(generated)          # [B, T]
        else:
            wave = vocoder(generated)                  # [B, 1, T] или [B, T]

        wave_np = wave.squeeze().detach().cpu().numpy().astype(np.float32)
        mel_np  = generated[0].detach().cpu().numpy()
        return wave_np, mel_np


# -------------------------- 3) СОХРАНЕНИЕ В WAV -------------------------- #
def save_wav(wave: np.ndarray, out_path: str, sr: int = 24000) -> str:
    """
    Сохраняет массив float32 [-1..1] в WAV и возвращает путь.
    """
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, wave, sr)
    return out_path