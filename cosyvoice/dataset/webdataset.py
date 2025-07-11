# lontou.2024
from pathlib import Path
import io
import json
import random
import functools

import webdataset as wds
import tarfile
import numpy as np

SPECIAL_TOKEN = "<|endofprompt|>"
PROB_INSTRUCTED = 1.0

name2url = {
    "azure": "gs://prod-ai-lab-speech-bucket/longtou/db/azure/wds_v2/shard-00000{0..7}.tar",
    "literature": "gs://prod-ai-lab-speech-bucket/longtou/db/literature/wds_v2/shard-0000{00..46}.tar",
    "skt_emotion_large": "gs://prod-ai-lab-speech-bucket/longtou/db/skt_emotion/wds_v2/large/shard-0000{00..24}.tar",
    "skt_emotion_small": "gs://prod-ai-lab-speech-bucket/longtou/db/skt_emotion/wds_v2/large/shard-0000{00..14}.tar",
    "mediazen_emotion": "gs://prod-ai-lab-speech-bucket/longtou/db/mediazen_emotion/wds_v2/shard-000{000..110}.tar",
    "mediazen": "gs://prod-ai-lab-speech-bucket/longtou/db/mediazen/wds_v2/shard-00{0000..1055}.tar",
    "commbooks": "gs://prod-ai-lab-speech-bucket/longtou/db/commbooks/wds_v2/shard-000{000..104}.tar",
    "kaist_audiobook": "gs://prod-ai-lab-speech-bucket/longtou/db/kaist_audiobook/wds_v2/shard-0000{00..10}.tar",
    "kaist_emotion": "gs://prod-ai-lab-speech-bucket/longtou/db/kaist_audiobook/wds_v2/shard-00000{0..8}.tar",
    "aihub_news": "gs://prod-ai-lab-speech-bucket/longtou/db/aihub_news/wds_v2/shard-000{000..107}.tar",
    "mediazen_adult": "gs://prod-ai-lab-speech-bucket/longtou/db/mediazen_adult/emilia_pipe/shard-0000{00..18}.tar",
    "mediazen_teen": "gs://prod-ai-lab-speech-bucket/longtou/db/mediazen_teen/emilia_pipe/shard-0000{00..11}.tar",
    "saltlux_jeju": "gs://prod-ai-lab-speech-bucket/longtou/db/saltlux_jeju/emilia_pipe/shard-00000{0..6}.tar",
    "saltlux_chungcheong": "gs://prod-ai-lab-speech-bucket/longtou/db/saltlux_chungcheong/emilia_pipe/shard-0000{00..17}.tar",
    "saltlux_gyeongsang": "gs://prod-ai-lab-speech-bucket/longtou/db/saltlux_gyeongsang/emilia_pipe/shard-0000{00..29}.tar",
    "saltlux_jeolla": "gs://prod-ai-lab-speech-bucket/longtou/db/saltlux_jeolla/emilia_pipe/shard-0000{00..20}.tar",
    "saltlux_gangwon": "gs://prod-ai-lab-speech-bucket/longtou/db/saltlux_gangwon/emilia_pipe/shard-0000{00..12}.tar",
    "emilia_en": "gs://prod-ai-lab-speech-bucket/longtou/db/emilia/wds/en/shard-00{0000..1092}.tar",
    "emilia_zh": "gs://prod-ai-lab-speech-bucket/longtou/db/emilia/wds/zh/shard-00{0000..1194}.tar",
    "emilia_ko": "gs://prod-ai-lab-speech-bucket/longtou/db/emilia/wds/ko/shard-00000{0..4}.tar",
    "emilia_yodas_ko": "gs://prod-ai-lab-speech-bucket/longtou/db/emilia_yodas/wds/ko/shard-000{000..207}.tar",
    "whispering": "gs://prod-ai-lab-speech-bucket/longtou/db/whispering/emilia_pipe/shard-000000.tar",
    "mediazen_teen_laugh": "gs://prod-ai-lab-speech-bucket/longtou/db/mediazen_teen/wds_laughter_tag/shard-000000.tar",
    "mediazen_adult_laugh": "gs://prod-ai-lab-speech-bucket/longtou/db/mediazen_adult/wds_laughter_tag/shard-000000.tar",
    "literature_speaking_rate": "gs://prod-ai-lab-speech-bucket/longtou/db/literature/wds_speaking_rate/shard-00000{0..2}.tar",
    "literature_tone": "gs://prod-ai-lab-speech-bucket/longtou/db/literature/wds_tone/shard-00000{0..2}.tar",
    "commbooks_speaking_rate": "gs://prod-ai-lab-speech-bucket/longtou/db/commbooks/wds_speaking_rate/shard-00000{0..5}.tar",
    "commbooks_tone": "gs://prod-ai-lab-speech-bucket/longtou/db/commbooks/wds_tone/shard-00000{0..7}.tar",
}

def PROMPT_TEMPLATE(spk_id, emotion, pitch, tone):
    # 화자, 성별, 감정, 피치, 톤, 속삭임, 나이,
    if pitch is None: pitch = ""
    if tone is None: tone = ""
    pitch = {"high": "높은음", "low": "낮은음", "": ""}[pitch]
    tone = {"dynamic": "다이나믹", "mono": "모노", "": ""}[tone]
    styles = [x for x in [emotion, pitch, tone] if x != ""]
    style = ""
    if len(styles) > 0:
        style = random.choice(styles)

    if random.random() < 0.5:
        spk_id = ""
    if random.random() < 0.5:
        style = ""

    prompt = f"화자는 {spk_id}, 스타일은 {style}"
    return prompt + SPECIAL_TOKEN

def SPK_PROMPT(spk_id):
    if random.random() < 0.5:
        return spk_id + SPECIAL_TOKEN
    else:
        return ""

### implement decoding function for each dataset ###

def decode_azure(sample):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["transcript"].strip()
    prompt = PROMPT_TEMPLATE(spk_id="azure", emotion="", pitch="", tone="")
    transcript = prompt + transcript
    return {"utt": uttid, "audio_data": wav, "text": transcript}

def decode_literature(sample, **kwargs):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["transcript"].strip()
    gender = json_data["gender"] # (MALE|FEMALE)
    spk_id = uttid.split('-')[2]

    emotion = ""
    # e.g. emotion) {'슬픔', '당황', '무감정', '불안', '상처', '기쁨', '분노'}
    emotion_style = json_data["emotion_style"]
    if len(emotion_style) > 0:
        #emotion_set = set()
        #style_set = set()
        #for item in emotion_style:
        #    emotion_set.add(item["emotion"])
        #    style_set.add(item["style"])
        emotion = emotion_style[0]["emotion"]

    prompt = PROMPT_TEMPLATE(spk_id=f"lit_{spk_id}", emotion=emotion, pitch="", tone="")
    transcript = prompt + transcript

    return {"utt": uttid, "audio_data": wav, "text": transcript}

def decode_literature_speaking_rate(sample, **kwargs):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text_tagged"].strip()
    gender = json_data["gender"] # (MALE|FEMALE)
    gender = "남자" if gender == "MALE" else "여자"
    spk_id = uttid.split('-')[2]

    emotion = ""
    # e.g. emotion) {'슬픔', '당황', '무감정', '불안', '상처', '기쁨', '분노'}
    emotion_style = json_data["emotion_style"]
    if len(emotion_style) > 0:
        #emotion_set = set()
        #style_set = set()
        #for item in emotion_style:
        #    emotion_set.add(item["emotion"])
        #    style_set.add(item["style"])
        emotion = emotion_style[0]["emotion"]

    prompt = PROMPT_TEMPLATE(spk_id=f"lit_{spk_id}", emotion="", pitch="", tone="")
    transcript = prompt + transcript

    return {"utt": uttid, "audio_data": wav, "text": transcript}

def decode_literature_tone(sample, **kwargs):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["transcript"].strip()
    gender = json_data["gender"] # (MALE|FEMALE)
    gender = "남자" if gender == "MALE" else "여자"
    spk_id = uttid.split('-')[2]
    pitch = json_data["pitch"]
    tone = json_data["tone"]

    emotion = ""
    # e.g. emotion) {'슬픔', '당황', '무감정', '불안', '상처', '기쁨', '분노'}
    emotion_style = json_data["emotion_style"]
    if len(emotion_style) > 0:
        #emotion_set = set()
        #style_set = set()
        #for item in emotion_style:
        #    emotion_set.add(item["emotion"])
        #    style_set.add(item["style"])
        emotion = emotion_style[0]["emotion"]

    prompt = PROMPT_TEMPLATE(spk_id=f"lit_{spk_id}", emotion="", pitch=pitch, tone=tone)
    transcript = prompt + transcript

    return {"utt": uttid, "audio_data": wav, "text": transcript}

def decode_mediazen(sample):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["전사정보"]["OrgLabelText"]

    spk_info = json_data["화자정보"]
    gender = spk_info["Gender"] # [Female|

    return {"utt": uttid, "audio_data": wav, "text": transcript}

def decode_skt_emotion_large(sample):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["transcript"]

    if random.random() < PROB_INSTRUCTED:
        # build instructed dataset if possible
        # e.g. style_main) {'SURPRISE', 'JOY', 'NEUTRAL', 'ANXIOUS', 'DOUBT', 'ANGRY', 'FEAR', 'KIND', 'SAD', 'HURRY', 'SERIOUS', 'DRY', 'SHY', 'UNPLEASURE', 'HESITATE', 'TEASE'}
        style_main = json_data["style_main"]
        #style_sub = json_data["style_sub"]
        prompt = style_main
        transcript = prompt + SPECIAL_TOKEN + transcript

    return {"utt": uttid, "audio_data": wav, "text": transcript}

def decode_skt_emotion_small(sample):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["transcript"]

    return {"utt": uttid, "audio_data": wav, "text": transcript}

def decode_mediazen_emotion(sample):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text_info"]["OrgLabelText"]

    # emotion: {'Happy', 'Sad', 'Anxious', 'Neutrality', 'Angry', 'Hurt', 'N/A', 'Embarrassed'}
    # sensitivity: {'자랑스럽다', '흐뭇하다', '섭섭하다', '아찔하다', ...
    # speech_style: {'뉴스체', '구연체', '대화체', '중계체', '낭독체', 'N/A'}
    # character: {'N/A', '아동', '일반', '노년'}
    # character emotion: {'N/A', '밝은', '어두운', '중립'}
    if random.random() < PROB_INSTRUCTED:
        # build instructed dataset if possible
        spk_info = json_data["spk_info"]
        emotion = spk_info["Emotion"]
        #_ = spk_info["Sensitivity"]
        speech_style = spk_info["SpeechStyle"]
        character = spk_info["Character"]
        character_emotion = spk_info["CharacterEmotion"]
        prompt = None
        if character in ('아동', '노년'):
            prompt = character
            if character_emotion in ('밝은', '어두운'):
                prompt = character_emotion + ' ' + prompt
        elif emotion in ('Happy', 'Sad', 'Anxious', 'Neutrality', 'Angry', 'Hurt', 'Embarrassed'):
            prompt = emotion
            if speech_style in ('뉴스체', '구연체', '대화체', '중계체', '낭독체'):
                prompt = prompt + " " + speech_style
        elif speech_style in ('뉴스체', '구연체', '대화체', '중계체', '낭독체'):
            prompt = speech_style

        if prompt is not None:
            transcript = prompt + SPECIAL_TOKEN + transcript

    return {"utt": uttid, "audio_data": wav, "text": transcript}

def decode_commbooks(sample, **kwargs):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    # tr: 십오 센티미터
    # origin_text: 15cm
    tr = json_data["voice_piece"]["tr"].strip()
    text = json_data["origin_text"].strip()
    if random.random() < 0.5:
        transcript = tr
    else:
        transcript = text
    gender = json_data["gender"]
    spk_id = uttid.split('-')[3]

    emotion = ""
    # emotion: {'기쁨', '무감정', '분노', '슬픔'}
    # intensity: {0, 1, 2, 3}
    # style: {'중계체', '대화체', '애니체', '낭독체', '친절체', '독백체', '구연체'}
    # sub_style: {'', '중학생', '20대 청년', '일반설명', '아빠', '40대,아저씨', '할머니', ...
    json_style = json_data["style"]
    this_emotion = json_style["emotion"]
    intensity = json_style["intensity"]
    style = json_style["style"]
    sub_style = json_style["sub_style"]
    if int(intensity) >= 2:
        emotion = this_emotion

    prompt = PROMPT_TEMPLATE(spk_id=f"cb_{spk_id}", emotion=emotion, pitch="", tone="")
    transcript = prompt + transcript

    return {"utt": uttid, "audio_data": wav, "text": transcript}

def decode_commbooks_speaking_rate(sample, **kwargs):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text_tagged"].strip()
    gender = json_data["gender"]
    spk_id = uttid.split('-')[3]

    emotion = ""
    # emotion: {'기쁨', '무감정', '분노', '슬픔'}
    # intensity: {0, 1, 2, 3}
    # style: {'중계체', '대화체', '애니체', '낭독체', '친절체', '독백체', '구연체'}
    # sub_style: {'', '중학생', '20대 청년', '일반설명', '아빠', '40대,아저씨', '할머니', ...
    json_style = json_data["style"]
    this_emotion = json_style["emotion"]
    intensity = json_style["intensity"]
    style = json_style["style"]
    sub_style = json_style["sub_style"]
    if int(intensity) >= 2:
        emotion = this_emotion

    prompt = PROMPT_TEMPLATE(spk_id=f"cb_{spk_id}", emotion="", pitch="", tone="")
    transcript = prompt + transcript

    return {"utt": uttid, "audio_data": wav, "text": transcript}

def decode_commbooks_tone(sample, **kwargs):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    # tr: 십오 센티미터
    # origin_text: 15cm
    tr = json_data["voice_piece"]["tr"].strip()
    text = json_data["origin_text"].strip()
    if random.random() < 0.5:
        transcript = tr
    else:
        transcript = text
    gender = json_data["gender"]
    spk_id = uttid.split('-')[3]
    pitch = json_data["pitch"]
    tone = json_data["tone"]

    emotion = ""
    # emotion: {'기쁨', '무감정', '분노', '슬픔'}
    # intensity: {0, 1, 2, 3}
    # style: {'중계체', '대화체', '애니체', '낭독체', '친절체', '독백체', '구연체'}
    # sub_style: {'', '중학생', '20대 청년', '일반설명', '아빠', '40대,아저씨', '할머니', ...
    json_style = json_data["style"]
    this_emotion = json_style["emotion"]
    intensity = json_style["intensity"]
    style = json_style["style"]
    sub_style = json_style["sub_style"]
    if int(intensity) >= 2:
        emotion = this_emotion

    prompt = PROMPT_TEMPLATE(spk_id=f"cb_{spk_id}", emotion="", pitch=pitch, tone=tone)
    transcript = prompt + transcript

    return {"utt": uttid, "audio_data": wav, "text": transcript}

def decode_saltlux_jeju(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"]

    if random.random() < PROB_INSTRUCTED:
        # build instructed dataset if possible
        prompt = "제주도 방언"
        transcript = prompt + SPECIAL_TOKEN + transcript

    return {"utt": uttid, "audio_data": mp3, "text": transcript}

def decode_saltlux_jeolla(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"]

    if random.random() < PROB_INSTRUCTED:
        # build instructed dataset if possible
        prompt = "전라도 방언"
        transcript = prompt + SPECIAL_TOKEN + transcript

    return {"utt": uttid, "audio_data": mp3, "text": transcript}

def decode_saltlux_chungcheong(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"]

    if random.random() < PROB_INSTRUCTED:
        # build instructed dataset if possible
        prompt = "충청도 방언"
        transcript = prompt + SPECIAL_TOKEN + transcript

    return {"utt": uttid, "audio_data": mp3, "text": transcript}

def decode_saltlux_gangwon(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"]

    if random.random() < PROB_INSTRUCTED:
        # build instructed dataset if possible
        prompt = "강원도 방언"
        transcript = prompt + SPECIAL_TOKEN + transcript

    return {"utt": uttid, "audio_data": mp3, "text": transcript}

def decode_saltlux_gyeongsang(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"]

    if random.random() < PROB_INSTRUCTED:
        # build instructed dataset if possible
        prompt = "경상도 방언"
        transcript = prompt + SPECIAL_TOKEN + transcript

    return {"utt": uttid, "audio_data": mp3, "text": transcript}

def decode_mediazen_adult(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"]

    return {"utt": uttid, "audio_data": mp3, "text": transcript}

def decode_mediazen_adult_laugh(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["laughter.json"]))
    transcript = json_data["transcript"]

    return {"utt": uttid, "audio_data": mp3, "text": transcript}

def decode_mediazen_teen(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"]

    return {"utt": uttid, "audio_data": mp3, "text": transcript}

def decode_mediazen_teen_laugh(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["laughter.json"]))
    transcript = json_data["transcript"]

    return {"utt": uttid, "audio_data": mp3, "text": transcript}

def decode_aihub_news(sample):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["script"]["transcript"]
    normalized = json_data["script"]["normalized"]
    if random.random() < 0.5:
        transcript = normalized

    if random.random() < PROB_INSTRUCTED:
        # build instructed dataset if possible
        prompt = "아나운서"
        transcript = prompt + SPECIAL_TOKEN + transcript

    return {"utt": uttid, "audio_data": wav, "text": transcript}

def decode_emilia_en(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"]

    return {"utt": uttid, "audio_data": mp3, "text": transcript}

def decode_emilia_zh(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"]

    return {"utt": uttid, "audio_data": mp3, "text": transcript}

def decode_emilia_ko(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"].strip()

    return {"utt": uttid, "audio_data": mp3, "text": transcript}

def decode_emilia_yodas_ko(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"].strip()

    return {"utt": uttid, "audio_data": mp3, "text": transcript}

def decode_whispering(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"].strip()

    prompt = "속삭임"
    transcript = prompt + SPECIAL_TOKEN + transcript

    return {"utt": uttid, "audio_data": mp3, "text": transcript}

### end of decoding function list ###


def build_wds(recipe_name, mode="train", cache_size=0, from_mount=False, from_prod=False):
    shard_url = name2url[recipe_name]
    if from_prod is False:
        shard_url = shard_url.replace("prod-", "")
        if from_mount is True:
            shard_url = shard_url.replace("gs://ai-lab-speech-bucket", "/home/longtou.2024/mount")
    else:
        if from_mount is True:
            shard_url = shard_url.replace("gs://prod-ai-lab-speech-bucket", "/home/longtou.2024/mount")
    cache_dir = None
    if cache_size > 0:
        cache_dir = f"wds_cache_{recipe_name}"
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    _decode = globals()[f"decode_{recipe_name}"]
    resampled = True if mode == "train" else False

    dataset = wds.WebDataset(
        shard_url,
        cache_dir=cache_dir,
        cache_size=cache_size,
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
        resampled=resampled, # if True, generate an infinite stream of samples
        shardshuffle=False, # ignored if resampled=True
        repeat=False,
    )

    dataset = dataset.map(_decode)
    return dataset

