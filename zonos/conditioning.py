import functools
from functools import cache, lru_cache
from typing import Any, Literal, Iterable

import torch
import torch.nn as nn

from zonos.config import PrefixConditionerConfig
from zonos.utils import DEFAULT_DEVICE

# Cache language code mapping to avoid repeated computation
@lru_cache(maxsize=1)
def _get_language_code_to_id_map() -> dict[str, int]:
    """Cache the language code to ID mapping to avoid repeated computation."""
    return {lang: i for i, lang in enumerate(supported_language_codes)}

# Cache commonly used tensor shapes and operations
@lru_cache(maxsize=128)
def _create_tensor_cache_key(value, device_type: str, dtype_str: str) -> str:
    """Create a cache key for tensor operations."""
    if isinstance(value, (list, tuple)):
        return f"{hash(tuple(value))}_{device_type}_{dtype_str}"
    return f"{hash(value)}_{device_type}_{dtype_str}"

class Conditioner(nn.Module):
    def __init__(
        self,
        output_dim: int,
        name: str,
        cond_dim: int | None = None,
        projection: Literal["none", "linear", "mlp"] = "none",
        uncond_type: Literal["learned", "none"] = "none",
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.output_dim = output_dim
        self.cond_dim = cond_dim = cond_dim or output_dim

        if projection == "linear":
            self.project = nn.Linear(cond_dim, output_dim)
        elif projection == "mlp":
            self.project = nn.Sequential(
                nn.Linear(cond_dim, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
            )
        else:
            self.project = nn.Identity()

        self.uncond_vector = None
        if uncond_type == "learned":
            self.uncond_vector = nn.Parameter(torch.zeros(output_dim))

    def apply_cond(self, *inputs: Any) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, inputs: tuple[Any, ...] | None) -> torch.Tensor:
        if inputs is None:
            assert self.uncond_vector is not None
            return self.uncond_vector.data.view(1, 1, -1)

        cond = self.apply_cond(*inputs)
        cond = self.project(cond)
        return cond


# ------- ESPEAK CONTAINMENT ZONE ------------------------------------------------------------------------------------------------------------------------------------------------
import os
import sys
import re
import unicodedata

import inflect
import torch
import torch.nn as nn
from kanjize import number2kanji
from phonemizer.backend import EspeakBackend
from sudachipy import Dictionary, SplitMode

if sys.platform == "darwin":
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/lib/libespeak-ng.dylib"
elif sys.platform == "win32":
    # Use Windows-style env var and expand it
    path1_with_env = r"%ProgramFiles%\\eSpeak NG"
    path2_with_env=r'%ProgramFiles%\\eSpeak NG\\libespeak-ng.dll'

    file_path1 = os.path.expandvars(path1_with_env)
    file_path2 = os.path.expandvars(path2_with_env)

    os.environ['PHONEMIZER_ESPEAK_PATH'] = file_path1
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = file_path2


# --- Number normalization code from https://github.com/daniilrobnikov/vits2/blob/main/text/normalize_numbers.py ---

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _remove_commas(m: re.Match) -> str:
    return m.group(1).replace(",", "")


def _expand_decimal_point(m: re.Match) -> str:
    return m.group(1).replace(".", " point ")


def _expand_dollars(m: re.Match) -> str:
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m: re.Match) -> str:
    return _inflect.number_to_words(m.group(0))


def _expand_number(m: re.Match) -> str:
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")


def normalize_numbers(text: str) -> str:
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


# --- Number normalization code end ---


PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKEN_IDS = [PAD_ID, UNK_ID, BOS_ID, EOS_ID]

_punctuation = ';:,.!?¡¿—…"«»“”() *~-/\\&'
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

symbols = [*_punctuation, *_letters, *_letters_ipa]
_symbol_to_id = {s: i for i, s in enumerate(symbols, start=len(SPECIAL_TOKEN_IDS))}


def _get_symbol_id(s: str) -> int:
    return _symbol_to_id.get(s, 1)


def get_symbol_ids(text: str) -> list[int]:
    return list(map(_get_symbol_id, text))


def tokenize_phonemes(phonemes: list[str]) -> tuple[torch.Tensor, list[int]]:
    phoneme_ids = [[BOS_ID, *get_symbol_ids(phonemes), EOS_ID] for phonemes in phonemes]
    lengths = list(map(len, phoneme_ids))
    longest = max(lengths)
    phoneme_ids = [[PAD_ID] * (longest - len(ids)) + ids for ids in phoneme_ids]
    return torch.tensor(phoneme_ids), lengths


# Cache Japanese tokenizer to avoid repeated initialization
@lru_cache(maxsize=1)
def _get_jp_tokenizer():
    """Cache Japanese tokenizer to avoid repeated initialization."""
    return Dictionary(dict="full").create()


def normalize_jp_text(text: str) -> str:
    """Optimized Japanese text normalization with cached tokenizer."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\d+", lambda m: number2kanji(int(m[0])), text)
    tokenizer = _get_jp_tokenizer()
    final_text = " ".join([x.reading_form() for x in tokenizer.tokenize(text, SplitMode.A)])
    return final_text


# Cache cleaned text results to avoid repeated processing
@lru_cache(maxsize=256)
def _clean_text_cached(text: str, language: str) -> str:
    """Cache cleaned text results to avoid repeated processing."""
    if "ja" in language:
        return normalize_jp_text(text)
    else:
        return normalize_numbers(text)


def clean(texts: list[str], languages: list[str]) -> list[str]:
    """Optimized clean function with caching."""
    return [_clean_text_cached(text, language) for text, language in zip(texts, languages)]


@cache
def get_backend(language: str) -> "EspeakBackend":
    import logging

    from phonemizer.backend import EspeakBackend

    logger = logging.getLogger("phonemizer")
    backend = EspeakBackend(
        language,
        preserve_punctuation=True,
        with_stress=True,
        punctuation_marks=_punctuation,
        logger=logger,
    )
    logger.setLevel(logging.ERROR)
    return backend


# Cache phonemization results to avoid repeated computation
@lru_cache(maxsize=512)
def _phonemize_single_cached(text: str, language: str) -> str:
    """Cache individual phonemization results to avoid repeated computation."""
    cleaned_text = _clean_text_cached(text, language)
    backend = get_backend(language)
    phonemes = backend.phonemize([cleaned_text], strip=True)
    return phonemes[0]


def phonemize(texts: list[str], languages: list[str]) -> list[str]:
    """Optimized phonemize function with caching."""
    return [_phonemize_single_cached(text, language) for text, language in zip(texts, languages)]


class EspeakPhonemeConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, **kwargs)
        self.phoneme_embedder = nn.Embedding(len(SPECIAL_TOKEN_IDS) + len(symbols), output_dim)

    def apply_cond(self, texts: list[str], languages: list[str]) -> torch.Tensor:
        """
        Args:
            texts: list of texts to convert to phonemes
            languages: ISO 639-1 -or otherwise eSpeak compatible- language code
        """
        device = self.phoneme_embedder.weight.device

        phonemes = phonemize(texts, languages)
        phoneme_ids, _ = tokenize_phonemes(phonemes)
        phoneme_embeds = self.phoneme_embedder(phoneme_ids.to(device))

        return phoneme_embeds


# ------- ESPEAK CONTAINMENT ZONE ------------------------------------------------------------------------------------------------------------------------------------------------


class FourierConditioner(Conditioner):
    def __init__(
        self,
        output_dim: int,
        input_dim: int = 1,
        std: float = 1.0,
        min_val: float = 0.0,
        max_val: float = 1.0,
        **kwargs,
    ):
        assert output_dim % 2 == 0
        super().__init__(output_dim, **kwargs)
        self.register_buffer("weight", torch.randn([output_dim // 2, input_dim]) * std)
        self.input_dim, self.min_val, self.max_val = input_dim, min_val, max_val

        # Cache normalization factor to avoid repeated computation
        self.register_buffer("_norm_factor", torch.tensor(self.max_val - self.min_val))

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_dim
        # Optimized normalization using cached factor
        x = (x - self.min_val) / self._norm_factor  # [batch_size, seq_len, input_dim]
        f = 2 * torch.pi * x.to(self.weight.dtype) @ self.weight.T  # [batch_size, seq_len, output_dim // 2]
        return torch.cat([f.cos(), f.sin()], dim=-1)  # [batch_size, seq_len, output_dim]


class IntegerConditioner(Conditioner):
    def __init__(self, output_dim: int, min_val: int = 0, max_val: int = 512, **kwargs):
        super().__init__(output_dim, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim)

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        # Handle input tensor shape - it might be [1, 1] or [batch_size, seq_len, 1]
        if x.dim() == 2:
            # Input is [batch_size, seq_len], need to add feature dimension
            x = x.unsqueeze(-1)  # Now [batch_size, seq_len, 1]

        assert x.shape[-1] == 1, f"Expected last dimension to be 1, got {x.shape[-1]}"

        # Apply embedding and ensure correct output shape
        embedded = self.int_embedder(x.squeeze(-1) - self.min_val)  # [batch_size, seq_len, output_dim]

        # Ensure we have the right shape [batch_size, 1, output_dim] for concatenation
        if embedded.dim() == 2:
            embedded = embedded.unsqueeze(1)  # Add sequence dimension

        return embedded


class PassthroughConditioner(Conditioner):
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(output_dim, **kwargs)

    def apply_cond(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.cond_dim
        return x


_cond_cls_map = {
    "PassthroughConditioner": PassthroughConditioner,
    "EspeakPhonemeConditioner": EspeakPhonemeConditioner,
    "FourierConditioner": FourierConditioner,
    "IntegerConditioner": IntegerConditioner,
}


def build_conditioners(conditioners: list[dict], output_dim: int) -> list[Conditioner]:
    return [_cond_cls_map[config["type"]](output_dim, **config) for config in conditioners]


class PrefixConditioner(Conditioner):
    def __init__(self, config: PrefixConditionerConfig, output_dim: int):
        super().__init__(output_dim, "prefix", projection=config.projection)
        self.conditioners = nn.ModuleList(build_conditioners(config.conditioners, output_dim))
        self.norm = nn.LayerNorm(output_dim)
        self.required_keys = {c.name for c in self.conditioners if c.uncond_vector is None}

        # Pre-allocate buffers for common batch sizes to avoid repeated allocations
        self._cached_shapes = {}
        self._max_cache_size = 8  # Limit cache size to prevent memory issues

    @lru_cache(maxsize=64)
    def _get_expansion_pattern(self, input_shape: tuple, target_batch_size: int) -> tuple:
        """Cache tensor expansion patterns to avoid repeated computation."""
        batch_size, seq_len, feature_dim = input_shape
        if batch_size == 1 and target_batch_size > 1:
            return (target_batch_size, seq_len, feature_dim)
        return input_shape

    def _ensure_3d_shape(self, tensor: torch.Tensor, target_batch_size: int = 1) -> torch.Tensor:
        """Efficiently ensure tensor has 3D shape with minimal operations."""
        if tensor.dim() == 1:
            # [feature_dim] -> [1, 1, feature_dim]
            return tensor.view(1, 1, -1)
        elif tensor.dim() == 2:
            # [batch_size, feature_dim] -> [batch_size, 1, feature_dim]
            return tensor.unsqueeze(1)
        elif tensor.dim() == 3:
            # Already correct shape, potentially expand batch dimension
            if tensor.shape[0] == 1 and target_batch_size > 1:
                return tensor.expand(target_batch_size, -1, -1)
            return tensor
        else:
            raise ValueError(f"Unexpected tensor dimension: {tensor.dim()}")

    def forward(self, cond_dict: dict) -> torch.Tensor:
        if not set(cond_dict).issuperset(self.required_keys):
            raise ValueError(f"Missing required keys: {self.required_keys - set(cond_dict)}")

        # Collect conditioner outputs first
        cond_outputs = []
        max_bsz = 1

        for conditioner in self.conditioners:
            cond_output = conditioner(cond_dict.get(conditioner.name))

            # Ensure output is at least 2D
            if cond_output.dim() == 1:
                cond_output = cond_output.unsqueeze(0)  # [1, feature_dim]

            # Track maximum batch size
            if cond_output.dim() >= 1:
                max_bsz = max(max_bsz, cond_output.shape[0])

            cond_outputs.append(cond_output)

        # Process outputs to ensure consistent 3D shape [batch, seq, features]
        processed_conds = []
        for cond_output in cond_outputs:
            # Convert to 3D: [batch_size, seq_len, feature_dim]
            if cond_output.dim() == 2:
                # [batch_size, feature_dim] -> [batch_size, 1, feature_dim]
                shaped_output = cond_output.unsqueeze(1)
            elif cond_output.dim() == 3:
                shaped_output = cond_output
            else:
                raise ValueError(f"Unexpected tensor dimension after initial processing: {cond_output.dim()}")

            # Expand batch dimension to match max_bsz if needed
            current_bsz = shaped_output.shape[0]
            if current_bsz == 1 and max_bsz > 1:
                shaped_output = shaped_output.expand(max_bsz, -1, -1)
            elif current_bsz != max_bsz:
                raise ValueError(f"Batch size mismatch: expected {max_bsz}, got {current_bsz}")

            processed_conds.append(shaped_output)

        # Verify all tensors have compatible shapes before concatenation
        if processed_conds:
            batch_size = processed_conds[0].shape[0]
            feature_dim = processed_conds[0].shape[2]

            for i, tensor in enumerate(processed_conds):
                if tensor.shape[0] != batch_size:
                    raise ValueError(f"Batch size mismatch at tensor {i}: expected {batch_size}, got {tensor.shape[0]}")
                if tensor.shape[2] != feature_dim:
                    raise ValueError(
                        f"Feature dimension mismatch at tensor {i}: expected {feature_dim}, got {tensor.shape[2]}")

        # Concatenate along sequence dimension
        concatenated = torch.cat(processed_conds, dim=1)  # Changed from dim=-2 to dim=1

        # Apply projection and normalization
        return self.norm(self.project(concatenated))

    def clear_shape_cache(self):
        """Clear the shape cache to free memory if needed."""
        self._get_expansion_pattern.cache_clear()
        self._cached_shapes.clear()


supported_language_codes = [
    'af', 'am', 'an', 'ar', 'as', 'az', 'ba', 'bg', 'bn', 'bpy', 'bs', 'ca', 'cmn',
    'cs', 'cy', 'da', 'de', 'el', 'en-029', 'en-gb', 'en-gb-scotland', 'en-gb-x-gbclan',
    'en-gb-x-gbcwmd', 'en-gb-x-rp', 'en-us', 'eo', 'es', 'es-419', 'et', 'eu', 'fa',
    'fa-latn', 'fi', 'fr-be', 'fr-ch', 'fr-fr', 'ga', 'gd', 'gn', 'grc', 'gu', 'hak',
    'hi', 'hr', 'ht', 'hu', 'hy', 'hyw', 'ia', 'id', 'is', 'it', 'ja', 'jbo', 'ka',
    'kk', 'kl', 'kn', 'ko', 'kok', 'ku', 'ky', 'la', 'lfn', 'lt', 'lv', 'mi', 'mk',
    'ml', 'mr', 'ms', 'mt', 'my', 'nb', 'nci', 'ne', 'nl', 'om', 'or', 'pa', 'pap',
    'pl', 'pt', 'pt-br', 'py', 'quc', 'ro', 'ru', 'ru-lv', 'sd', 'shn', 'si', 'sk',
    'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'tn', 'tr', 'tt', 'ur', 'uz', 'vi',
    'vi-vn-x-central', 'vi-vn-x-south', 'yue'
]  # fmt: off

def make_cond_dict(
    text: str = "It would be nice to have time for testing, indeed.",
    language: str = "en-us",
    speaker: torch.Tensor | None = None,
    
    # Emotion vector from 0.0 to 1.0
    #   Is entangled with pitch_std because more emotion => more pitch variation
    #                     VQScore and DNSMOS because they favor neutral speech
    #
    #                       Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral
    emotion: list[float] = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077],

    # Maximum frequency (0 to 24000), should be 22050 or 24000 for 44.1 or 48 kHz audio
    # For voice cloning use 22050
    fmax: float = 22050.0,
    
    # Standard deviation for pitch (0 to 400), should be 
    #   20-45 for normal speech, 
    #   60-150 for expressive speech, 
    #   higher values => crazier samples
    pitch_std: float = 20.0,

    # Speaking rate in phonemes per minute (0 to 40). 30 is very fast, 10 is slow.
    speaking_rate: float = 15.0,

    # Target VoiceQualityScore for the generated speech (0.5 to 0.8).
    #   A list of values must be provided which represent each 1/8th of the audio.
    #   You should unset for expressive speech.
    # According to discord Chat this is only used for the hybrid model
    vqscore_8: list[float] = [0.78] * 8,

    # CTC target loss
    # Only used for the hybrid model
    ctc_loss: float = 0.0,
    # Only used for the hybrid model
    dnsmos_ovrl: float = 4.0,
    # Only used for the hybrid model
    speaker_noised: bool = False,
    unconditional_keys: Iterable[str] = {"vqscore_8", "dnsmos_ovrl"},
    device: torch.device | str = DEFAULT_DEVICE,
) -> dict:
    """
    A helper to build the 'cond_dict' that the model expects.
    By default, it will generate a random speaker embedding

    Optimized version with reduced GPU-to-CPU copies and caching.
    """
    assert language.lower() in supported_language_codes, "Please pick a supported language"

    # Use cached language code mapping
    language_code_to_id = _get_language_code_to_id_map()

    cond_dict = {
        "espeak": ([text], [language]),
        "speaker": speaker,
        "emotion": emotion,
        "fmax": fmax,
        "pitch_std": pitch_std,
        "speaking_rate": speaking_rate,
        "language_id": language_code_to_id[language],
        "vqscore_8": vqscore_8,
        "ctc_loss": ctc_loss,
        "dnsmos_ovrl": dnsmos_ovrl,
        "speaker_noised": int(speaker_noised),
    }

    # Remove unconditional keys early
    for k in unconditional_keys:
        cond_dict.pop(k, None)

    # Optimize tensor operations - reduce device transfers and conversions
    target_device = torch.device(device) if isinstance(device, str) else device

    # Define which keys should be treated as integers (for IntegerConditioner)
    integer_keys = {"language_id", "speaker_noised"}

    # Process all non-tensor values efficiently
    for k, v in cond_dict.items():
        if isinstance(v, (float, int, list)):
            if isinstance(v, list):
                # Create tensor directly on target device with appropriate dtype
                tensor_val = torch.tensor(v, device=target_device, dtype=torch.float32)
            else:
                # Handle integer vs float types appropriately
                if k in integer_keys:
                    # For integer conditioners, use long dtype and don't reshape to (1,1,-1)
                    tensor_val = torch.tensor(v, device=target_device, dtype=torch.long)
                    cond_dict[k] = tensor_val.view(1, 1)  # Shape: [1, 1] for integer embeddings
                    continue
                else:
                    # Create scalar tensor directly on target device
                    tensor_val = torch.tensor(v, device=target_device, dtype=torch.float32)

            # Reshape efficiently - avoid multiple operations
            cond_dict[k] = tensor_val.view(1, 1, -1)

        elif isinstance(v, torch.Tensor):
            # Minimize device transfers - only move if necessary
            if v.device != target_device:
                v = v.to(target_device)

            # Handle integer tensors appropriately
            if k in integer_keys and v.dtype.is_floating_point:
                v = v.long()  # Convert to integer type
                cond_dict[k] = v.view(1, 1)  # Shape: [1, 1] for integer embeddings
            else:
                cond_dict[k] = v.view(1, 1, -1)

    # Normalize emotion efficiently if present
    if "emotion" in cond_dict:
        emotion_tensor = cond_dict["emotion"]
        # Normalize in-place to avoid creating additional tensors
        emotion_sum = emotion_tensor.sum(dim=-1, keepdim=True)
        cond_dict["emotion"] = emotion_tensor / emotion_sum

    return cond_dict


# Cache management utilities for debugging and monitoring
def get_conditioning_cache_stats() -> dict:
    """Get statistics about conditioning caches."""
    return {
        "phonemize_cache_size": _phonemize_single_cached.cache_info().currsize,
        "phonemize_cache_hits": _phonemize_single_cached.cache_info().hits,
        "phonemize_cache_misses": _phonemize_single_cached.cache_info().misses,
        "text_clean_cache_size": _clean_text_cached.cache_info().currsize,
        "text_clean_cache_hits": _clean_text_cached.cache_info().hits,
        "text_clean_cache_misses": _clean_text_cached.cache_info().misses,
        "backend_cache_size": len(get_backend.__wrapped__.__cache__),
    }


def clear_conditioning_caches():
    """Clear all conditioning caches."""
    _phonemize_single_cached.cache_clear()
    _clean_text_cached.cache_clear()
    get_backend.cache_clear()
    _get_language_code_to_id_map.cache_clear()
    _get_jp_tokenizer.cache_clear()
