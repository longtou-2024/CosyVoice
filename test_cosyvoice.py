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

#ref_audio_dir = "ref_audio_all/ref_audio"
ref_audio_dir = "ref_audio_demo"
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

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)
# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage

for uttid in scp.keys():
    if uttid == "LEEBYEONGHEON-01":
        continue
    fpath = scp[uttid]
    prompt_speech_16k = load_wav(fpath, 16000)
    prompt_text = text_raw[uttid].strip()
    for idx, sent in enumerate(sent_syn):
        tts_text = sent.strip()
        tts_speech = []
        #for tts_output in cosyvoice.inference_zero_shot((x for x in ["안녕하세요? 보이는 라디오 시청자 여러분~", "저는 진행자 정지영입니다. 오늘 하루는 어떠신가요? [laughter]"]), prompt_text, prompt_speech_16k, stream=True):
        for tts_output in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False):
            tts_speech.append(tts_output['tts_speech'])
            #print(tts_speech.shape) # [C,T]
        if len(tts_speech) != 1:
            tts_speech = torch.cat(tts_speech, dim=1) # [C,T]
        else:
            tts_speech = tts_speech[0]
        torchaudio.save(outdir / f'{uttid}_zs_{idx}.wav', tts_speech, cosyvoice.sample_rate)


# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
#for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#    torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
