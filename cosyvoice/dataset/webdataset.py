# lontou.2024
from pathlib import Path
import io
import json
import random

import webdataset as wds

SPECIAL_TOKEN = "<|endofprompt|>"
PROB_INSTRUCTED = 0.1

name2url = {
    "azure": "gs://ai-lab-speech-bucket/longtou/db/azure/wds_v2/shard-00000{0..7}.tar",
    #"literature": "gs://ai-lab-speech-bucket/longtou/db/literature/wds_v2/shard-0000{00..46}.tar",
    "literature": "gs://ai-lab-speech-bucket/longtou/db/literature/wds_v2/shard-0000{00..02}.tar",
    "skt_emotion_large": "gs://ai-lab-speech-bucket/longtou/db/skt_emotion/wds_v2/large/shard-0000{00..24}.tar",
    "skt_emotion_small": "gs://ai-lab-speech-bucket/longtou/db/skt_emotion/wds_v2/large/shard-0000{00..14}.tar",
    "mediazen_emotion": "gs://ai-lab-speech-bucket/longtou/db/mediazen_emotion/wds_v2/shard-000{000..110}.tar",
    "mediazen": "gs://ai-lab-speech-bucket/longtou/db/mediazen/wds_v2/shard-00{0000..1055}.tar",
    "commbooks": "gs://ai-lab-speech-bucket/longtou/db/commbooks/wds_v2/shard-000{000..104}.tar",
    "kaist_audiobook": "gs://ai-lab-speech-bucket/longtou/db/kaist_audiobook/wds_v2/shard-0000{00..10}.tar",
    "kaist_emotion": "gs://ai-lab-speech-bucket/longtou/db/kaist_audiobook/wds_v2/shard-00000{0..8}.tar",
    "aihub_news": "gs://ai-lab-speech-bucket/longtou/db/aihub_news/wds_v2/shard-000{000..107}.tar",
    "mediazen_adult": "gs://ai-lab-speech-bucket/longtou/db/mediazen_adult/emilia_pipe/shard-0000{00..18}.tar",
    "mediazen_teen": "gs://ai-lab-speech-bucket/longtou/db/mediazen_teen/emilia_pipe/shard-0000{00..11}.tar",
    "saltlux_jeju": "gs://ai-lab-speech-bucket/longtou/db/saltlux_jeju/emilia_pipe/shard-00000{0..6}.tar",
    "saltlux_chungcheong": "gs://ai-lab-speech-bucket/longtou/db/saltlux_chungcheong/emilia_pipe/shard-0000{00..17}.tar",
    "saltlux_gyeongsang": "gs://ai-lab-speech-bucket/longtou/db/saltlux_gyeongsang/emilia_pipe/shard-0000{00..29}.tar",
    "saltlux_jeolla": "gs://ai-lab-speech-bucket/longtou/db/saltlux_jeolla/emilia_pipe/shard-0000{00..20}.tar",
    "saltlux_gangwon": "gs://ai-lab-speech-bucket/longtou/db/saltlux_gangwon/emilia_pipe/shard-0000{00..12}.tar",
    "emilia_en": "gs://ai-lab-speech-bucket/longtou/db/emilia/wds/en/shard-00{0000..1092}.tar",
    "emilia_zh": "gs://ai-lab-speech-bucket/longtou/db/emilia/wds/zh/shard-00{0000..1194}.tar",
    "emilia_ko": "gs://ai-lab-speech-bucket/longtou/db/emilia/wds/ko/shard-00000{0..4}.tar",
    "emilia_yodas_ko": "gs://ai-lab-speech-bucket/longtou/db/emilia_yodas/wds/ko/shard-000{000..207}.tar",
}

### implement decoding function for each dataset ###

def decode_azure(sample):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["transcript"]
    return {"utt": uttid, "audio_data": wav, "text": transcript}

def decode_literature(sample):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["transcript"]

    if random.random() < PROB_INSTRUCTED:
        # build instructed dataset if possible
        # e.g. emotion) {'슬픔', '당황', '무감정', '불안', '상처', '기쁨', '분노'}
        emotion_style = json_data["emotion_style"]
        if len(emotion_style) > 0:
            #emotion_set = set()
            #style_set = set()
            #for item in emotion_style:
            #    emotion_set.add(item["emotion"])
            #    style_set.add(item["style"])

            prompt = emotion_style[0]["emotion"]
            transcript = prompt + SPECIAL_TOKEN + transcript

    return {"utt": uttid, "audio_data": wav, "text": transcript}

def decode_mediazen(sample):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["전사정보"]["OrgLabelText"]
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

def decode_commbooks(sample):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    # tr: 십오 센티미터
    # origin_text: 15cm
    tr = json_data["voice_piece"]["tr"]
    text = json_data["origin_text"]
    if random.random() < 0.5:
        transcript = tr
    else:
        transcript = text

    if random.random() < PROB_INSTRUCTED:
        # build instructed dataset if possible
        # emotion: {'기쁨', '무감정', '분노', '슬픔'}
        # intensity: {0, 1, 2, 3}
        # style: {'중계체', '대화체', '애니체', '낭독체', '친절체', '독백체', '구연체'}
        # sub_style: {'', '중학생', '20대 청년', '일반설명', '아빠', '40대,아저씨', '할머니', ...
        json_style = json_data["style"]
        emotion = json_style["emotion"]
        intensity = json_style["intensity"]
        style = json_style["style"]
        sub_style = json_style["sub_style"]
        prompt = ""
        if emotion in ('기쁨', '무감정', '분노', '슬픔'):
            prompt += emotion
        if style in ('중계체', '애니체', '낭독체', '친절체', '독백체', '구연체'):
            prompt += f" {style}"
        if prompt != "":
            transcript = prompt.strip() + SPECIAL_TOKEN + transcript

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

def decode_mediazen_teen(sample):
    uttid = sample["__key__"]
    mp3 = sample["mp3"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["text"]

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

### end of decoding function list ###


def build_wds(recipe_name, mode="train"):
    shard_url = name2url[recipe_name]
    cache_dir = f"wds_cache_{recipe_name}"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    _decode = globals()[f"decode_{recipe_name}"]
    resampled = True if mode == "train" else False
    dataset = wds.WebDataset(
        shard_url,
        cache_dir=cache_dir,
        cache_size=int(1e9),
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
        resampled=resampled, # if True, generate an infinite stream of samples
        shardshuffle=False, # ignored if resampled=True
        repeat=False,
    )

    dataset = dataset.map(_decode)
    return dataset

