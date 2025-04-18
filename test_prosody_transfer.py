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

prosody_audio_dir = "prosody_audio_dir"
#ref_audio_dir = "ref_audio_all/ref_audio"
ref_audio_dir = "ref_audio_demo"
pr_scp = {}
for line in open(prosody_audio_dir + "/wav.scp", 'r').readlines():
    uttid, fpath = line.strip().split(' ', maxsplit=1)
    pr_scp[uttid] = fpath
pr_text_raw = {}
for line in open(prosody_audio_dir + "/text_raw", 'r').readlines():
    uttid, line = line.strip().split(' ', maxsplit=1)
    pr_text_raw[uttid] = line
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

# load model
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)

# prepare prosody prompt
uttid = next(iter(pr_scp.keys()))
fpath = pr_scp[uttid]
pr_prompt_speech_16k = load_wav(fpath, 16000)
pr_prompt_text = pr_text_raw[uttid]

for uttid in scp.keys():
    if uttid == "LEEBYEONGHEON-01":
        continue
    fpath = scp[uttid]
    prompt_speech_16k = load_wav(fpath, 16000)
    prompt_text = text_raw[uttid].strip()
    for idx, sent in enumerate(sent_syn):
        tts_text = sent.strip()
        tts_speech = []
        for tts_output in cosyvoice.inference_zero_shot_prosody_transfer("아니 또 이러시네! 자꾸 이러면 나 더이상 못해요~", pr_prompt_text, pr_prompt_speech_16k, prompt_text, prompt_speech_16k, stream=False):
            tts_speech.append(tts_output['tts_speech'])
            #print(tts_speech.shape) # [C,T]
        if len(tts_speech) != 1:
            tts_speech = torch.cat(tts_speech, dim=1) # [C,T]
        else:
            tts_speech = tts_speech[0]
        torchaudio.save(outdir / f'{uttid}_zs_{idx}.wav', tts_speech, cosyvoice.sample_rate)

