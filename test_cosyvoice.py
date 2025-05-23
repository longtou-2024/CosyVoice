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

ref_audio_dir = "ref_azure"


cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)

#for role in ["ke_narrator", "ke_kim", "ke_park"]:
for role in [""]:
    dname = ref_audio_dir + f"/{role}"
    sent_syn = open(dname + "/sent_syn.txt", 'r').readlines()

    outdir = Path(f"outdir/{role}")
    outdir.mkdir(parents=True, exist_ok=True)

    text_raw = {}
    for line in open(dname + "/text", 'r').readlines():
        uttid, line = line.strip().split(' ', maxsplit=1)
        text_raw[uttid] = line
    #text_raw = open(dname + "/text", 'r').readlines()
    #assert len(text_raw) == 1

    audio_path_list = list(Path(dname).glob("*.wav"))

    for audio_path in audio_path_list:
        uttid = Path(audio_path).stem
        prompt_speech_16k = load_wav(audio_path, 16000)
        prompt_text = text_raw[uttid].strip()
        #prompt_text = text_raw[0].strip()
        for idx, sent in enumerate(sent_syn):
            tts_text = sent.strip()
            tts_speech = []
            for tts_output in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False):
                tts_speech.append(tts_output['tts_speech'])
            tts_speech = torch.concat(tts_speech, dim=1) # [C,T]
            torchaudio.save(outdir / f'{uttid}_zs_{idx}.wav', tts_speech, cosyvoice.sample_rate)
