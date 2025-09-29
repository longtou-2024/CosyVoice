import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.') # for cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pathlib import Path
import json
import torch


spk_roles = {
    "m_narrator": ["N0217-01-21-00", "5042_G1A2E7_KJB_004142", "5039_G1A2E7_KIM_000872", "1491_G1A2E7_JJW_001089"],
    "f_narrator": ["9035_G2A1E7_PHS_001420"],
    "father": ["M-NX-D-005-0301"],
    "soldier": ["N0245-02-29-00", "7569_G1A2E7_KJK_003070", "N0001-06-29-00", "N0231-09-29-08"],
    "male": ["N0204-10-31-00", "D-NX-F-003-0201", "1658_G1A2E7_HMK_001674", "0042_G1A3E7S4C0_CHS_001786"],
    "female": ["N0169-05-30-00", "9027_G2A1E7_HJS_001223", "9013_G2A1E7_PYE_000237", "5817_G2A2E7_KSI_002989", "4503_G2A2E7_LMN_003875", "5817_G2A2E7_KSI_002783"],
    "female_child": ["A-NX-D-010-0051", "0033_G2A3E7S0C2_KMA_001625" ,"0033_G2A3E2S0C3_KMA_001680"],
}

celect_script = [
    {
        "tag": "매우 강한 기쁨",
        "text": '와~ 드디어 우리 게임 출시됐어! 오늘은 밤새 파티다!',
    },
    {
        "tag": "매우 강한 슬픔",
        "text": '그때 그렇게 말하지 말 걸... 아직도 마음에 남아.',
    },
    {
        "tag": "매우 강한 걱정",
        "text": '혹시 답장 아직 안 온 거예요? 뭔가 문제가 있는 건 아닌지...',
    },
    {
        "tag": "매우 강한 화난",
        "text": '아니, 왜 자꾸 그 사람 얘기만 하는 거야?',
    },
    {
        "tag": "매우 강한 무감정",
        "text": '그날 이후, 둘은 다시 연락하지 않았다.',
    },
    {
        "tag": "매우 강한 당황",
        "text": '아, 그 얘긴 그냥... 농담이었어요. 진짜예요...',
    },
    {
        "tag": "매우 강한 슬픔",
        "text": '주변엔 늘 사람들이 있는데... 왜 이렇게 외로울까.',
    },
    {
        "tag": "매우 강한 화난",
        "text": '그만 좀 해! 말했잖아, 난 그런 거 관심 없다고!',
    },
    {
        "tag": "매우 강한 기쁨",
        "text": '비 온 뒤에 무지개가 떴다. 마음 한편이 따뜻해졌다.',
    },
    {
        "tag": "매우 강한 걱정",
        "text": '혹시... 내가 이상하게 보이진 않았죠? 다들 쳐다보는 것 같아서...',
    },
]

script_2 = [
    {'role': 'heroine', 'style': '절망한듯', 'text': '이거, 꿈이 아니잖아!, 내 삶을 돌려달라고!'},
    {'role': 'heroine', 'style': '혼란스러운', 'text': '현실을 부정하며 발버둥 치던 중, 한 병사가 다가와 나를 살피기 시작했다'},
    {'role': 'soldier', 'style': '궁금한듯', 'text': '“흐음... 얘가 왜 이러지? 아픈가? 보고를 올려야 하나?”'},
    {'role': 'heroine', 'style': '안도하는', 'text': '보고...? 일단 나는 이 사람의 새는 아닌가 보군!'},
    {'role': 'heroine', 'style': '단호하게', 'text': '이대로 잡혀있을 순 없지!'},
    {'role': 'heroine', 'style': '조심스러운', 'text': '나는 죽은 척 연기하며 기회를 노렸고,'},
    {'role': 'heroine', 'style': '용맹스럽게', 'text': '이내 발톱 맛을 보여주며 탈출했다'},
    {'role': 'heroine', 'style': '신난듯', 'text': '성공했어!, 내가 해냈다고!'},
    {'role': 'heroine', 'style': '궁금한듯', 'text': '신나게 날아다니다 숲에 도착했는데...'},
    {'role': 'heroine', 'style': '충격받은', 'text': '우리 왕국의 깃발과... 사람...?!'},
    {'role': 'heroine', 'style': '걱정스러운', 'text': '쓰러진 아군 전령병을 발견했고,'},
    {'role': 'heroine', 'style': '비장한', 'text': '그의 품에는 왕국의 운명이 걸린 기밀문이 있었다'},
    {'role': 'heroine', 'style': '다급하게', 'text': '게다가, 심장이 뛰잖아...?!, 이대로 두면 죽을 거야...!'},
    {'role': 'heroine', 'style': '책임감있는듯', 'text': '나는 왕녀니까, 병사를 구하고 문서를 전달해야만 해!'},
    {'role': 'heroine', 'style': '다급하게', 'text': '필사적으로 찾아 헤맨 끝에 마침내 아군진지를 발견했고,'},
    {'role': 'heroine', 'style': '다급하게', 'text': '나는 그곳을 향해 전속력으로 돌진했다'},
    {'role': 'heroine', 'style': '당황한듯', 'text': '그런데 잠깐, 어... 어떻게 멈추는 거더라...?'},
    {'role': 'heroine', 'style': '충격받은', 'text': '속도 조절에 실패한 나는 결국 한 남자에게 부딪히기 직전,'},
    {'role': 'heroine', 'style': '놀란듯', 'text': '그의 손에 붙잡히고 말았는데...'},
    {'role': 'hero', 'style': '냉정하게', 'text': '“짐승도 암살에 쓰나?”'},
]


def prefix_tag(tag, text):
    return f"<|tag_start|>{tag}<|tag_end|>{text}"

def wrap_tag(tag, text):
    split = text.split(' ')
    tagged = []
    for word in split:
        tagged.append(f"<{tag}>{word}</{tag}>")
    return " ".join(tagged)

llm_path="/home/longtou.2024/mount/longtou/h100/exp/cosyvoice/20250926/torch_ddp/epoch_1_step_26000.pt"
flow_path=None
cosyvoice = CosyVoice2('/home/longtou.2024/mount/longtou/saved/cosyvoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False, llm_path=llm_path, flow_path=flow_path)

outdir = "outdir"
Path(outdir).mkdir(exist_ok=True)

# kast_emo
#prompt_spk_speech_16k = dict()
#for audio_path in Path("test/kast_emo").glob("*.wav"):
#    uttid = audio_path.stem
#    prompt_spk_speech_16k[uttid] = load_wav(audio_path, 16000)
#
#prompt_spk_speech_24k = dict()
#for audio_path in Path("test/kast_emo").glob("*.wav"):
#    uttid = audio_path.stem
#    prompt_spk_speech_24k[uttid] = load_wav(audio_path, 24000)
#
#prompt_spk_sent = dict()
#for line in open("test/kast_emo/text", 'r').readlines():
#    uttid, transcript = line.split(' ', maxsplit=1)
#    prompt_spk_sent[uttid] = transcript

# qpp
prompt_spk_speech_16k = dict()
for audio_path in Path("test/qpp2").glob("*.wav"):
    uttid = audio_path.stem
    prompt_spk_speech_16k[uttid] = load_wav(audio_path, 16000)
prompt_spk_speech_24k = dict()
for audio_path in Path("test/qpp2").glob("*.wav"):
    uttid = audio_path.stem
    prompt_spk_speech_24k[uttid] = load_wav(audio_path, 24000)

prompt_spk_sent = dict()
for json_path in Path("test/qpp2").glob("*.json"):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    uttid = json_path.stem
    if "transcript" in json_data:
        transcript = json_data["transcript"].strip()
    elif "voice_piece" in json_data:
        transcript = json_data["voice_piece"]["tr"].strip()
    elif "text_info" in json_data:
        transcript = json_data["text_info"]["OrgLabelText"].strip()
    elif "전사정보" in json_data:
        transcript = json_data["전사정보"]["OrgLabelText"].strip()
    else:
        raise Exception
    prompt_spk_sent[uttid] = transcript


# shorts
sh_prompt_spk_speech_16k = dict()
for audio_path in Path("test/shorts/shorts_wavs_lt").glob("**/*.wav"):
    uttid = audio_path.stem
    sh_prompt_spk_speech_16k[uttid] = load_wav(audio_path, 16000)
sh_prompt_spk_speech_24k = dict()
for audio_path in Path("test/shorts/shorts_wavs_lt").glob("**/*.wav"):
    uttid = audio_path.stem
    sh_prompt_spk_speech_24k[uttid] = load_wav(audio_path, 24000)

sh_prompt_spk_sent = dict()
for json_path in Path("test/shorts/shorts_wavs_lt").glob("**/*.json"):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    uttid = json_path.stem
    if "transcript" in json_data:
        transcript = json_data["transcript"].strip()
    elif "voice_piece" in json_data:
        transcript = json_data["voice_piece"]["tr"].strip()
    elif "text_info" in json_data:
        transcript = json_data["text_info"]["OrgLabelText"].strip()
    elif "전사정보" in json_data:
        transcript = json_data["전사정보"]["OrgLabelText"].strip()
    else:
        raise Exception
    sh_prompt_spk_sent[uttid] = transcript

# select shorts spk
uttid = spk_roles["female"][5]
sh_prompt_sent = sh_prompt_spk_sent[uttid]
sh_prompt_speech_16k = sh_prompt_spk_speech_16k[uttid]
sh_prompt_speech_24k = sh_prompt_spk_speech_24k[uttid]

for d_idx, spk_id in enumerate(prompt_spk_sent):
    prompt_sent = prompt_spk_sent[spk_id]
    prompt_speech_16k = prompt_spk_speech_16k[spk_id]
    prompt_speech_24k = prompt_spk_speech_24k[spk_id]

    prompt_speech_16k = torch.cat([prompt_speech_16k, sh_prompt_speech_16k], dim=1)
    prompt_speech_24k = torch.cat([prompt_speech_24k, sh_prompt_speech_24k], dim=1)

    prompt_text = prefix_tag("prosody", prompt_sent)
    prompt_text += prefix_tag("timbre", sh_prompt_sent)

    tts_speech = []
    for t_idx, item in enumerate(script_2):
        this_text = prefix_tag("transfer", item["text"])
        gen = cosyvoice.inference_zero_shot(this_text, prompt_text, prompt_speech_16k, stream=False, text_frontend=False, prompt_speech_24k=prompt_speech_24k)
        ret = next(gen)
        tts_speech.append(ret['tts_speech'])
    tts_speech = torch.cat(tts_speech, dim=1)
    torchaudio.save(f"{outdir}/{spk_id}_{uttid}.wav", tts_speech, cosyvoice.sample_rate)

