# lontou.2024
from pathlib import Path
import io
import json
import random
import functools

import webdataset as wds
import tarfile
import numpy as np

ENDOFPROMPT = "<|endofprompt|>"
TAG_START = "<|tag_start|>"
TAG_END = "<|tag_end|>"
#PROB_INSTRUCTED = 1.0

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
    "mediazen_adult": "gs://prod-ai-lab-speech-bucket/longtou/db/mediazen_adult/emilia_pipe_v2/shard-000{{000..010},{100..110}}.tar",
    "mediazen_teen": "gs://prod-ai-lab-speech-bucket/longtou/db/mediazen_teen/emilia_pipe_v2/shard-000{{000..005},{100..105}}.tar",
    "saltlux_jeju": "gs://prod-ai-lab-speech-bucket/longtou/db/saltlux_jeju/emilia_pipe/shard-00000{0..6}.tar",
    "saltlux_chungcheong": "gs://prod-ai-lab-speech-bucket/longtou/db/saltlux_chungcheong/emilia_pipe/shard-0000{00..17}.tar",
    "saltlux_gyeongsang": "gs://prod-ai-lab-speech-bucket/longtou/db/saltlux_gyeongsang/emilia_pipe/shard-0000{00..29}.tar",
    "saltlux_jeolla": "gs://prod-ai-lab-speech-bucket/longtou/db/saltlux_jeolla/emilia_pipe/shard-0000{00..20}.tar",
    "saltlux_gangwon": "gs://prod-ai-lab-speech-bucket/longtou/db/saltlux_gangwon/emilia_pipe/shard-0000{00..12}.tar",
    "emilia_en": "gs://prod-ai-lab-speech-bucket/longtou/db/emilia/wds/en/shard-00{0000..1092}.tar",
    "emilia_zh": "gs://prod-ai-lab-speech-bucket/longtou/db/emilia/wds/zh/shard-00{0000..1194}.tar",
    "emilia_ko": "gs://prod-ai-lab-speech-bucket/longtou/db/emilia/wds/ko/shard-00000{0..4}.tar",
    "emilia_yodas_en": "gs://prod-ai-lab-speech-bucket/longtou/db/emilia_yodas/wds/en/shard-00{0000..1361}.tar",
    "emilia_yodas_ko": "gs://prod-ai-lab-speech-bucket/longtou/db/emilia_yodas/wds/ko/shard-000{000..207}.tar",
    "whispering": "gs://prod-ai-lab-speech-bucket/longtou/db/whispering/emilia_pipe/shard-000000.tar",
    "mediazen_teen_laugh": "gs://prod-ai-lab-speech-bucket/longtou/db/mediazen_teen/wds_laughter_tag/shard-000000.tar",
    "mediazen_adult_laugh": "gs://prod-ai-lab-speech-bucket/longtou/db/mediazen_adult/wds_laughter_tag/shard-000000.tar",
    "literature_speaking_rate": "gs://prod-ai-lab-speech-bucket/longtou/db/literature/wds_speaking_rate/shard-00000{0..2}.tar",
    "literature_tone": "gs://prod-ai-lab-speech-bucket/longtou/db/literature/wds_tone/shard-00000{0..2}.tar",
    "commbooks_speaking_rate": "gs://prod-ai-lab-speech-bucket/longtou/db/commbooks/wds_speaking_rate/shard-00000{0..5}.tar",
    "commbooks_tone": "gs://prod-ai-lab-speech-bucket/longtou/db/commbooks/wds_tone/shard-00000{0..7}.tar",
    "ke_youtube": "gs://prod-ai-lab-speech-bucket/longtou/db/ke_youtube/emilia_pipe/shard-0000{00..25}.tar",
    "ke_youtube2": "gs://prod-ai-lab-speech-bucket/longtou/db/ke_youtube2/emilia_pipe/shard-0000{00..19}.tar",
    "ke_youtube3": "gs://prod-ai-lab-speech-bucket/longtou/db/ke_youtube3/emilia_pipe/shard-000{{000..031},{100..129}}.tar",
}

### implement decoding function for each dataset ###

def decode_azure(sample):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["transcript"].strip()

    #prompt = PROMPT_TEMPLATE(spk="azure", style=None, mask_probs=[0,0])
    transcript = "애저 화자" + ENDOFPROMPT + transcript
    return {"utt": uttid, "audio_data": wav, "text": transcript, "spk_id": "unkown", "tag": "unkown"}

def decode_literature(sample, **kwargs):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["transcript"].strip()
    gender = json_data["gender"] # (MALE|FEMALE)
    gender = {"MALE": 'M', "FEMALE": 'F'}[gender]
    spk_id = f"lit_{uttid.split('-')[2]}"

    emotion = "무감정"
    # e.g. emotion) {'슬픔', '당황', '무감정', '불안', '상처', '기쁨', '분노'}
    emotion_style = json_data["emotion_style"]
    if len(emotion_style) > 0:
        #emotion_set = set()
        #style_set = set()
        #for item in emotion_style:
        #    emotion_set.add(item["emotion"])
        #    style_set.add(item["style"])
        emotion = emotion_style[0]["emotion"]

    tag = {"슬픔": "sad", "당황": "embarrassed", "무감정": "neutral", "불안": "anxious", "상처": "hurt", "기쁨": "happy", "분노": "angry"}[emotion]
    #transcript = TAG_START + tag + TAG_END + transcript
    tag = TAG_START + tag + TAG_END

    return {"utt": uttid, "audio_data": wav, "text": transcript, "spk_id": spk_id, "tag": tag}

def decode_literature_speaking_rate(sample, **kwargs):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text_tagged"].strip()
    gender = json_data["gender"] # (MALE|FEMALE)
    gender = "남자" if gender == "MALE" else "여자"
    spk_id = f"lit_{uttid.split('-')[2]}"

    emotion = "무감정"
    # e.g. emotion) {'슬픔', '당황', '무감정', '불안', '상처', '기쁨', '분노'}
    emotion_style = json_data["emotion_style"]
    if len(emotion_style) > 0:
        #emotion_set = set()
        #style_set = set()
        #for item in emotion_style:
        #    emotion_set.add(item["emotion"])
        #    style_set.add(item["style"])
        emotion = emotion_style[0]["emotion"]

    if random.random() < 0.5:
        # without tagging
        tag = "unkown"
        spk_id = "unkown"
    else:
        tag = {"슬픔": "sad", "당황": "embarrassed", "무감정": "neutral", "불안": "anxious", "상처": "hurt", "기쁨": "happy", "분노": "angry"}[emotion]
        #transcript = TAG_START + tag + TAG_END + transcript
        tag = TAG_START + tag + TAG_END

    return {"utt": uttid, "audio_data": wav, "text": transcript, "spk_id": spk_id, "tag": tag}

def decode_literature_tone(sample, **kwargs):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["transcript"].strip()
    gender = json_data["gender"] # (MALE|FEMALE)
    gender = {"MALE": 'm', "FEMALE": 'f'}[gender]
    spk_id = f"lit_{uttid.split('-')[2]}"

    pitch = json_data["pitch"]
    tone = json_data["tone"]

    if pitch is None:
        style = tone
    elif tone is None:
        style = pitch
    else:
        style = random.choice([pitch, tone])
    tag = f"{style}"
    #transcript = TAG_START + tag + TAG_END + transcript
    tag = TAG_START + tag + TAG_END

    return {"utt": uttid, "audio_data": wav, "text": transcript, "spk_id": spk_id, "tag": tag}

def decode_mediazen(sample):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["전사정보"]["OrgLabelText"].strip()

    spk_info = json_data["화자정보"]
    gender = spk_info["Gender"] # [Female|


    return {"utt": uttid, "audio_data": wav, "text": transcript, "spk_id": "unkown", "tag": "unkown"}

def decode_skt_emotion_large(sample):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["transcript"]
    spk_id = f"skt_{uttid.split('_')[0]}"

    style_main_set = {'SURPRISE', 'JOY', 'NEUTRAL', 'ANXIOUS', 'DOUBT', 'ANGRY', 'FEAR', 'KIND', 'SAD', 'HURRY', 'SERIOUS', 'DRY', 'SHY', 'UNPLEASURE', 'HESITATE', 'TEASE'}
    style_main = json_data["style_main"]
    #style_sub = json_data["style_sub"]
    # NOTE(longtou): fix typo
    fix_typo = {"SY": "SHY", "ESITATE": "HESITATE", "NEUTRA": "NEUTRAL", "URRY": "HURRY", "UNPEASURE": "UNPLEASURE"}
    if style_main in fix_typo:
        style_main = fix_typo[style_main]
    assert style_main in style_main_set
    tag = style_main.lower()

    #transcript = TAG_START + tag + TAG_END + transcript
    tag = TAG_START + tag + TAG_END

    return {"utt": uttid, "audio_data": wav, "text": transcript, "spk_id": spk_id, "tag": tag}

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
    gender = {"MALE": '남성', "FEMALE": '여성'}[gender]
    spk_id = f"cb_{uttid.split('-')[3]}"

    # emotion: {'기쁨', '무감정', '분노', '슬픔'}
    # intensity: {0, 1, 2, 3}
    # style: {'중계체', '대화체', '애니체', '낭독체', '친절체', '독백체', '구연체'}
    # sub_style: {'', '중학생', '20대 청년', '일반설명', '아빠', '40대,아저씨', '할머니', ...
    json_style = json_data["style"]
    emotion = json_style["emotion"]
    intensity = json_style["intensity"]
    style = json_style["style"]
    sub_style = json_style["sub_style"]
    #if int(intensity) >= 2:
    #    emotion = this_emotion

    if emotion == "무감정":
        tag = style
    else:
        tag = {"기쁨": "happy", "분노": "angry", "슬픔": "sad"}[emotion]
        tag = f"{tag} {intensity}"

    #transcript = TAG_START + tag + TAG_END + transcript
    tag = TAG_START + tag + TAG_END

    return {"utt": uttid, "audio_data": wav, "text": transcript, "spk_id": spk_id, "tag": tag}

def decode_commbooks_speaking_rate(sample, **kwargs):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text_tagged"].strip()
    gender = json_data["gender"]
    gender = {"MALE": 'M', "FEMALE": 'F'}[gender]
    spk_id = f"cb_{uttid.split('-')[3]}"

    json_style = json_data["style"]
    emotion = json_style["emotion"]
    intensity = json_style["intensity"]
    style = json_style["style"]
    sub_style = json_style["sub_style"]
    #if int(intensity) >= 2:
    #    emotion = this_emotion

    if random.random() < 0.5:
        # without tagging
        tag = "unkown"
        spk_id = "unkown"
    else:
        if emotion == "무감정":
            tag = style
        else:
            tag = {"기쁨": "happy", "분노": "angry", "슬픔": "sad"}[emotion]
            tag = f"{tag} {intensity}"
        #transcript = TAG_START + tag + TAG_END + transcript
        tag = TAG_START + tag + TAG_END

    return {"utt": uttid, "audio_data": wav, "text": transcript, "spk_id": spk_id, "tag": tag}

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
    gender = {"MALE": 'm', "FEMALE": 'f'}[gender]
    spk_id = f"cb_{uttid.split('-')[3]}"
    pitch = json_data["pitch"]
    tone = json_data["tone"]

    if pitch is None:
        style = tone
    elif tone is None:
        style = pitch
    else:
        style = random.choice([pitch, tone])

    tag = f"{style}"
    #transcript = TAG_START + tag + TAG_END + transcript
    tag = TAG_START + tag + TAG_END

    return {"utt": uttid, "audio_data": wav, "text": transcript, "spk_id": spk_id, "tag": tag}

def decode_saltlux_jeju(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"].strip()

    return {"utt": uttid, "audio_data": mp3, "text": transcript, "spk_id": "unkown", "tag": "unkown"}

def decode_saltlux_jeolla(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"].strip()

    return {"utt": uttid, "audio_data": mp3, "text": transcript, "spk_id": "unkown", "tag": "unkown"}

def decode_saltlux_chungcheong(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"].strip()

    return {"utt": uttid, "audio_data": mp3, "text": transcript, "spk_id": "unkown", "tag": "unkown"}

def decode_saltlux_gangwon(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"].strip()

    return {"utt": uttid, "audio_data": mp3, "text": transcript, "spk_id": "unkown", "tag": "unkown"}

def decode_saltlux_gyeongsang(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"].strip()

    return {"utt": uttid, "audio_data": mp3, "text": transcript, "spk_id": "unkown", "tag": "unkown"}

def decode_mediazen_adult(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"]
    spk_id = f"ma_{uttid.rsplit('_', maxsplit=1)[0]}"

    tag = "chat adult"
    #transcript = TAG_START + tag + TAG_END + transcript
    tag = TAG_START + tag + TAG_END

    return {"utt": uttid, "audio_data": mp3, "text": transcript, "spk_id": spk_id, "tag": tag}

def decode_mediazen_adult_laugh(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["laughter.json"]))
    transcript = json_data["transcript"]
    spk_id = f"ma_{uttid.rsplit('_', maxsplit=1)[0]}"

    if random.random() < 0.5:
        tag = "unkown"
        spk_id = "unkown"
    else:
        tag = "chat adult"
        #transcript = TAG_START + tag + TAG_END + transcript
        tag = TAG_START + tag + TAG_END

    return {"utt": uttid, "audio_data": mp3, "text": transcript, "spk_id": spk_id, "tag": tag}

def decode_mediazen_teen(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"]
    spk_id = f"mt_{uttid.rsplit('_', maxsplit=1)[0]}"

    tag = "chat teen"
    #transcript = TAG_START + tag + TAG_END + transcript
    tag = TAG_START + tag + TAG_END

    return {"utt": uttid, "audio_data": mp3, "text": transcript, "spk_id": spk_id, "tag": tag}

def decode_mediazen_teen_laugh(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["laughter.json"]))
    transcript = json_data["transcript"]
    spk_id = f"mt_{uttid.rsplit('_', maxsplit=1)[0]}"

    if random.random() < 0.5:
        tag = "unkown"
        spk_id = "unkown"
    else:
        tag = "chat teen"
        #transcript = TAG_START + tag + TAG_END + transcript
        tag = TAG_START + tag + TAG_END

    return {"utt": uttid, "audio_data": mp3, "text": transcript, "spk_id": spk_id, "tag": tag}

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

    return {"utt": uttid, "audio_data": mp3, "text": transcript, "spk_id": "unkown", "tag": "unkown"}

def decode_emilia_yodas_en(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"].strip()

    return {"utt": uttid, "audio_data": mp3, "text": transcript, "spk_id": "unkown", "tag": "unkown"}

def decode_emilia_yodas_ko(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"].strip()

    return {"utt": uttid, "audio_data": mp3, "text": transcript, "spk_id": "unkown", "tag": "unkown"}

def decode_whispering(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"].strip()

    transcript = "속삭임" + ENDOFPROMPT + transcript

    return {"utt": uttid, "audio_data": mp3, "text": transcript, "spk_id": "unkown", "tag": "unkown"}

def decode_ke_youtube(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"].strip()

    return {"utt": uttid, "audio_data": mp3, "text": transcript, "spk_id": "unkown", "tag": "unkown"}

def decode_ke_youtube2(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"].strip()

    return {"utt": uttid, "audio_data": mp3, "text": transcript, "spk_id": "unkown", "tag": "unkown"}

def decode_ke_youtube3(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"].strip()

    return {"utt": uttid, "audio_data": mp3, "text": transcript, "spk_id": "unkown", "tag": "unkown"}

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
    # NOTE(longtou): if valid, avoid infinite loop
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

    dataset = dataset.shuffle(1000)
    dataset = dataset.map(_decode)
    return dataset

