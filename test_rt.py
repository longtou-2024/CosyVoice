## SDK模型下载
#from modelscope import snapshot_download
#snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')

import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pathlib import Path
import torch

ref_audio_dir = "ref_audio_all/ref_audio"
#ref_audio_dir = "ref_audio_demo"
scp = {}
for line in open(ref_audio_dir + "/wav.scp", 'r').readlines():
    uttid, fpath = line.strip().split(' ', maxsplit=1)
    scp[uttid] = fpath
text_raw = {}
for line in open(ref_audio_dir + "/text_raw", 'r').readlines():
    uttid, line = line.strip().split(' ', maxsplit=1)
    text_raw[uttid] = line
sent_syn = open(ref_audio_dir + "/sent_syn.txt", 'r').readlines()

outdir = Path("outdir")
outdir.mkdir(parents=True, exist_ok=True)

# NOTE(longtou):
# RTF
# - base, fp16, jit, stream
stream = True

use_flow_cache = False
if stream is True:
    use_flow_cache = True # avoid OOM

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=use_flow_cache)

def text_gen():
    yield "퍼플렉시티를 스마트폰에 탑재할 경우, 기본 에이아이 어시스턴트 옵션으로 제공하거나 퍼플렉시티 안드로이드 앱을 휴대전화에 사전 설치하는 방식이 채택될 전망이다."
def text_fn():
    return "퍼플렉시티를 스마트폰에 탑재할 경우, 기본 에이아이 어시스턴트 옵션으로 제공하거나 퍼플렉시티 안드로이드 앱을 휴대전화에 사전 설치하는 방식이 채택될 전망이다."

for uttid in scp.keys():
    if uttid == "LEEBYEONGHEON-01":
        continue
    fpath = scp[uttid]
    prompt_speech_16k = load_wav(fpath, 16000)
    prompt_text = text_raw[uttid].strip()
    for idx, sent in enumerate(sent_syn):
        tts_text = sent.strip()
        tts_speech = []
        for tts_output in cosyvoice.inference_zero_shot_rt(text_gen(), prompt_text, prompt_speech_16k, stream=stream, text_frontend=False):
            tts_speech.append(tts_output['tts_speech'])
            #print(tts_speech.shape) # [C,T]
        if len(tts_speech) != 1:
            tts_speech = torch.cat(tts_speech, dim=1) # [C,T]
        else:
            tts_speech = tts_speech[0]
        torchaudio.save(outdir / f'{uttid}_zs_{idx}.wav', tts_speech, cosyvoice.sample_rate)
        print(f"{uttid} saved")

