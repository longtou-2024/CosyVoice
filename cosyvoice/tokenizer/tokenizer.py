import base64
import os
from functools import lru_cache
from typing import Optional
import torch
from transformers import AutoTokenizer
from whisper.tokenizer import Tokenizer
import onnxruntime

import tiktoken


LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
    "minnan": "minnan",
    "wuyu": "wuyu",
    "dialect": "dialect",
    "zh/en": "zh/en",
    "en/zh": "en/zh",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}

AUDIO_EVENT = {
    "ASR": "ASR",
    "AED": "AED",
    "SER": "SER",
    "Speech": "Speech",
    "/Speech": "/Speech",
    "BGM": "BGM",
    "/BGM": "/BGM",
    "Laughter": "Laughter",
    "/Laughter": "/Laughter",
    "Applause": "Applause",
    "/Applause": "/Applause",
}

EMOTION = {
    "HAPPY": "HAPPY",
    "SAD": "SAD",
    "ANGRY": "ANGRY",
    "NEUTRAL": "NEUTRAL",
}

TTS_Vocal_Token = {
    "TTS/B": "TTS/B",
    "TTS/O": "TTS/O",
    "TTS/Q": "TTS/Q",
    "TTS/A": "TTS/A",
    "TTS/CO": "TTS/CO",
    "TTS/CL": "TTS/CL",
    "TTS/H": "TTS/H",
    **{f"TTS/SP{i:02d}": f"TTS/SP{i:02d}" for i in range(1, 14)}
}


@lru_cache(maxsize=None)
def get_encoding(name: str = "gpt2", num_languages: int = 99):
    vocab_path = os.path.join(os.path.dirname(__file__), "assets", f"{name}.tiktoken")
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path) if line)
    }
    n_vocab = len(ranks)
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        *[f"<|{audio_event}|>" for audio_event in list(AUDIO_EVENT.keys())],
        *[f"<|{emotion}|>" for emotion in list(EMOTION.keys())],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 31)],        # register special tokens for ASR
        *[f"<|{tts}|>" for tts in list(TTS_Vocal_Token.keys())],  # register special tokens for TTS
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


@lru_cache(maxsize=None)
def get_tokenizer(
    multilingual: bool,
    *,
    num_languages: int = 99,
    language: Optional[str] = None,
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]
) -> Tokenizer:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if multilingual:
        encoding_name = "multilingual_zh_ja_yue_char_del"
        language = language or "en"
        task = task or "transcribe"
    else:
        encoding_name = "gpt2"
        language = None
        task = None

    encoding = get_encoding(name=encoding_name, num_languages=num_languages)

    return Tokenizer(
        encoding=encoding, num_languages=num_languages, language=language, task=task
    )


class QwenTokenizer():
    def __init__(self, token_path, skip_special_tokens=True):
        super().__init__()
        # NOTE: non-chat model, all these special tokens keep randomly initialized.
        special_tokens = {
            'eos_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'additional_special_tokens': [
                '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]',
                "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]",
                "<fast>", "</fast>", "<slow>", "</slow>",
                "<|tag_start|>", "<|tag_end|>"
            ]
        }
        # e.g. emotion) {'슬픔', '당황', '무감정', '불안', '상처', '기쁨', '분노'}
        self.special_tokens = special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        self.tokenizer.add_special_tokens(special_tokens)
        self.skip_special_tokens = skip_special_tokens

    def encode(self, text, **kwargs):
        tokens = self.tokenizer([text], return_tensors="pt")
        tokens = tokens["input_ids"][0].cpu().tolist()
        return tokens

    def decode(self, tokens):
        tokens = torch.tensor(tokens, dtype=torch.int64)
        text = self.tokenizer.batch_decode([tokens], skip_special_tokens=self.skip_special_tokens)[0]
        return text


@lru_cache(maxsize=None)
def get_qwen_tokenizer(
    token_path: str,
    skip_special_tokens: bool
) -> QwenTokenizer:
    return QwenTokenizer(token_path=token_path, skip_special_tokens=skip_special_tokens)

@lru_cache(maxsize=None)
def get_campplus_session(
    onnx_path
):
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(onnx_path, sess_options=option, providers=providers)
    return ort_session


CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

supports = {
    # CAM++ trained on 200k labeled speakers
    'iic/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
}

@lru_cache(maxsize=None)
def get_campplus_model(
):
    #import sys
    #sys.path.append('3D-Speaker')
    from speakerlab.utils.builder import dynamic_import
    conf = supports['iic/speech_campplus_sv_zh-cn_16k-common']
    #cache_dir = snapshot_download(
    #            args.model_id,
    #            revision=conf['revision'],
    #            )
    #cache_dir = pathlib.Path(cache_dir)
    cache_dir = "/home/longtou.2024/mount/longtou/saved/cosyvoice/pretrained_models/speech_campplus_sv_zh-cn_16k-common"
    pretrained_model = f"{cache_dir}/{conf['model_pt']}"
    pretrained_state = torch.load(pretrained_model, map_location='cpu')

    model = conf['model']
    embedding_model = dynamic_import(model['obj'])(**model['args'])
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.to(torch.device("cpu"))
    embedding_model.eval()
    return embedding_model
