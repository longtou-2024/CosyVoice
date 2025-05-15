## SDK模型下载
#from modelscope import snapshot_download
#snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')

import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pathlib import Path
import torch

ref_audio_dir = "ref_audio_pej"
prosody_audio_dir = "ref_prosody_cosy"

# parse ref
scp = {}
for line in Path(ref_audio_dir).glob("*.wav"):
    fpath = str(line)
    uttid = line.stem
    scp[uttid] = fpath
text_raw = {}
for line in open(ref_audio_dir + "/text_raw", 'r').readlines():
    uttid, line = line.strip().split(' ', maxsplit=1)
    text_raw[uttid] = line

# parse prosody
pr_scp = {}
for line in Path(prosody_audio_dir).glob("*.wav"):
    fpath = str(line)
    uttid = line.stem
    pr_scp[uttid] = fpath
pr_text_raw = {}
for line in open(prosody_audio_dir + "/text_raw", 'r').readlines():
    uttid, line = line.strip().split(' ', maxsplit=1)
    pr_text_raw[uttid] = line


# load model
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)

outdir = Path("outdir")
outdir.mkdir(parents=True, exist_ok=True)

for uttid in scp.keys():
    fpath = scp[uttid]
    prompt_speech_16k = load_wav(fpath, 16000)
    prompt_text = text_raw[uttid].strip()

    for pr_uttid in  pr_scp.keys():
        pr_fpath = pr_scp[pr_uttid]
        pr_prompt_speech_16k = load_wav(pr_fpath, 16000)
        pr_prompt_text = pr_text_raw[pr_uttid]
        prosody = pr_uttid.split('_', maxsplit=1)[0]

        tts_speeches = []
        #def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
        for tts_output in cosyvoice.inference_vc(pr_prompt_speech_16k, prompt_speech_16k, stream=False):
            tts_speeches.append(tts_output['tts_speech'])
        tts_speeches = torch.concat(tts_speeches, dim=1) # [C,T]
        torchaudio.save(outdir / f'{uttid}_{prosody}.wav', tts_speeches, cosyvoice.sample_rate)
    break
