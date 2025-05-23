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

ref_audio_dir = "dialect/azure"
prosody_audio_dir = "dialect"

# load model
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)


# parse ref
audio_path_list = list(Path(ref_audio_dir).glob("*.wav"))
assert len(audio_path_list) == 1
audio_path = audio_path_list[0]
prompt_speech_16k = load_wav(audio_path, 16000)
#text_path = str(Path(audio_path).parent) + "/text"
#lines = open(text_path, 'r').readlines()
#assert len(lines) == 0
#pr_prompt_text = lines[0].strip().split(' ', maxsplit=1)[1]


for role in ["jeju"]:
    dname = prosody_audio_dir + f"/{role}"

    outdir = Path(f"outdir/{role}")
    outdir.mkdir(parents=True, exist_ok=True)

    audio_path_list = list(Path(dname).glob("*.wav"))
    assert len(audio_path_list) == 1
    audio_path = audio_path_list[0]

    uttid = Path(audio_path).stem
    pr_prompt_speech_16k = load_wav(audio_path, 16000)

    tts_speeches = []
    for tts_output in cosyvoice.inference_vc(pr_prompt_speech_16k, prompt_speech_16k, stream=False):
        tts_speeches.append(tts_output['tts_speech'])
    tts_speeches = torch.concat(tts_speeches, dim=1) # [C,T]
    torchaudio.save(outdir / f'{uttid}_{role}.wav', tts_speeches, cosyvoice.sample_rate)
