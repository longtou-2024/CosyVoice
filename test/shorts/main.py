import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.') # for cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pathlib import Path
import json, io
import torch

spk_roles = {
    "m_narrator": ["N0217-01-21-00", "5042_G1A2E7_KJB_004142", "5039_G1A2E7_KIM_000872", "1491_G1A2E7_JJW_001089"],
    "f_narrator": ["9035_G2A1E7_PHS_001420"],
    "father": ["M-NX-D-005-0301"],
    "soldier": ["N0245-02-29-00", "7569_G1A2E7_KJK_003070"],
    "male": ["N0204-10-31-00", "D-NX-F-003-0201", "1658_G1A2E7_HMK_001674", "0042_G1A3E7S4C0_CHS_001786"],
    "female": ["N0169-05-30-00", "9027_G2A1E7_HJS_001223", "9013_G2A1E7_PYE_000237", "5817_G2A2E7_KSI_002989", "4503_G2A2E7_LMN_003875"],
    "female_child": ["A-NX-D-010-0051", "0033_G2A3E7S0C2_KMA_001625" ,"0033_G2A3E2S0C3_KMA_001680"],
}

script_1 = [
    {'role': 'heroine', 'style': '근엄하게', 'text': '긍지 높던 황금의 왕국, 로이몬드'},
    {'role': 'heroine', 'style': '근엄하게', 'text': '어느 날... 왕의 과욕으로 인해 찬란했던 영광은 사라지고 한순간에 무너져내렸는데...'},
    {'role': 'soldier', 'style': '호통치듯', 'text': '“잡아라!”'},
    {'role': 'heroine', 'style': '다급하게', 'text': '적들의 맹렬한 추격 속 죽기살기로 도망치고 있는 나, 제르이네'},
    {'role': 'heroine', 'style': '근엄하게', 'text': '나는 왕국의 여덟 번째 왕녀다'},
    {'role': 'heroine', 'style': '괴로운듯', 'text': '언젠가 왕국을 다시 일으키려면, 후계자인 언니들이 꼭 살아남아야 하는데...'},
    {'role': 'heroine', 'style': '걱정', 'text': '이를 위해선, 내가 미끼가 되어야겠지...'},
    {'role': 'heroine', 'style': '서러운듯', 'text': '하지만 나는… 살고 싶어! 이렇게 허무하게 죽을 순 없어!'},
    {'role': 'heroine', 'style': '다급하게', 'text': '그 순간...'},
    {'role': 'soldier', 'style': '급하다', 'text': '“왕녀님! 조심…!”'},
]


def prefix_tag(tag, text):
    return f"<|tag_start|>{tag}<|tag_end|>{text}"

#llm_path="/home/longtou.2024/projects/CosyVoice/examples/aihub/cosyvoice2/exp/cosyvoice2/llm/torch_ddp/llm_avg.pt"
llm_path="/home/longtou.2024/mount/longtou/h100/exp/cosyvoice/20250908/torch_ddp/epoch_1_step_80000.pt"
cosyvoice = CosyVoice2('/home/longtou.2024/mount/longtou/saved/cosyvoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False, llm_path=llm_path)

outdir = "outdir"
Path(outdir).mkdir(exist_ok=True)

prompt_spk_speech_16k = dict()
for audio_path in Path("test/shorts/shorts_wavs_lt").glob("*.wav"):
    uttid = audio_path.stem
    prompt_spk_speech_16k[uttid] = load_wav(audio_path, 16000)

prompt_spk_sent = dict()
for json_path in Path("test/shorts/shorts_wavs_lt").glob("*.json"):
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

female_num = 3
male_num = 0
soldier_num = 1
tts_speech = []
tts_uttid = set()
for x in script_1:
    if x['role'] == 'heroine':
        uttid = spk_roles['female'][female_num]
    elif x['role'] == 'hero':
        uttid = spk_roles['male'][male_num]
    elif x['role'] == 'soldier':
        uttid = spk_roles['soldier'][soldier_num]
    else:
        raise Exception(x['role'])
    prompt_sent = prompt_spk_sent[uttid]
    prompt_speech_16k = prompt_spk_speech_16k[uttid]

    tag = x["style"]
    this_text = x["text"]
    #for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag(tag, this_text), prompt_sent, prompt_speech_16k, stream=False, text_frontend=False)):
    if tag == '':
        gen = cosyvoice.inference_zero_shot(f" {this_text}", prompt_sent, prompt_speech_16k, stream=False, text_frontend=False)
    else:
        gen = cosyvoice.inference_zero_shot(prefix_tag(tag, this_text), prompt_sent, prompt_speech_16k, stream=False, text_frontend=False)
    ret = next(gen)
    tts_speech.append(ret['tts_speech'])
    tts_uttid.add(uttid)
tts_speech = torch.cat(tts_speech, dim=1)
wav_name = '@'.join(tts_uttid)
torchaudio.save(f"{outdir}/{wav_name}.wav", tts_speech, cosyvoice.sample_rate)

