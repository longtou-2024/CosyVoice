import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.') # for cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pathlib import Path

from cosyvoice.dataset.webdataset import PROMPT_TEMPLATE

llm_path="/home/longtou.2024/projects/CosyVoice/examples/aihub/cosyvoice2/exp/cosyvoice2/llm/torch_ddp/llm_avg.pt"
cosyvoice = CosyVoice2('/home/longtou.2024/mount/longtou/saved/cosyvoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False, llm_path=llm_path)

outdir = "outdir"
Path(outdir).mkdir(exist_ok=True)
prompt_speech_16k = load_wav('./test/azure/azure-01.wav', 16000)
#prompt_speech_16k = load_wav('./test/ke_kim/F-A3-D-005-0051.wav', 16000)
spk_id="azure"

fout_names = ["base.wav", "fast.wav", "slow.wav", "happy.wav", "sad.wav",
              "angry.wav", "high.wav", "low.wav", "dynamic.wav", "mono.wav" ]


for i, j in enumerate(cosyvoice.inference_instruct2('어느 맑은 봄날, 바람에 이리저리 휫날리는 나뭇가지를 바라보며 제자가 물었다. 스승님, 저것은 나뭇가지가 움직이는 겁니까? 바람이 움직이는 겁니까?', PROMPT_TEMPLATE(spk_id=spk_id, emotion="", pitch="", tone="", infer=True), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2('<fast>어느 맑은 봄날, 바람에 이리저리 휫날리는 나뭇가지를 바라보며 제자가 물었다. 스승님, 저것은 나뭇가지가 움직이는 겁니까? 바람이 움직이는 겁니까?</fast>', PROMPT_TEMPLATE(spk_id=spk_id, emotion="", pitch="", tone="", infer=True), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2('<slow>어느 맑은 봄날, 바람에 이리저리 휫날리는 나뭇가지를 바라보며 제자가 물었다. 스승님, 저것은 나뭇가지가 움직이는 겁니까? 바람이 움직이는 겁니까?</slow>', PROMPT_TEMPLATE(spk_id=spk_id, emotion="", pitch="", tone="", infer=True), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2('어느 맑은 봄날, 바람에 이리저리 휫날리는 나뭇가지를 바라보며 제자가 물었다. 스승님, 저것은 나뭇가지가 움직이는 겁니까? 바람이 움직이는 겁니까?', PROMPT_TEMPLATE(spk_id=spk_id, emotion="기쁨", pitch="", tone="", infer=True), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[3]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2('어느 맑은 봄날, 바람에 이리저리 휫날리는 나뭇가지를 바라보며 제자가 물었다. 스승님, 저것은 나뭇가지가 움직이는 겁니까? 바람이 움직이는 겁니까?', PROMPT_TEMPLATE(spk_id=spk_id, emotion="슬픔", pitch="", tone="", infer=True), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[4]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2('어느 맑은 봄날, 바람에 이리저리 휫날리는 나뭇가지를 바라보며 제자가 물었다. 스승님, 저것은 나뭇가지가 움직이는 겁니까? 바람이 움직이는 겁니까?', PROMPT_TEMPLATE(spk_id=spk_id, emotion="분노", pitch="", tone="", infer=True), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[5]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2('어느 맑은 봄날, 바람에 이리저리 휫날리는 나뭇가지를 바라보며 제자가 물었다. 스승님, 저것은 나뭇가지가 움직이는 겁니까? 바람이 움직이는 겁니까?', PROMPT_TEMPLATE(spk_id=spk_id, emotion="", pitch="high", tone="", infer=True), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[6]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2('어느 맑은 봄날, 바람에 이리저리 휫날리는 나뭇가지를 바라보며 제자가 물었다. 스승님, 저것은 나뭇가지가 움직이는 겁니까? 바람이 움직이는 겁니까?', PROMPT_TEMPLATE(spk_id=spk_id, emotion="", pitch="low", tone="", infer=True), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[7]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2('어느 맑은 봄날, 바람에 이리저리 휫날리는 나뭇가지를 바라보며 제자가 물었다. 스승님, 저것은 나뭇가지가 움직이는 겁니까? 바람이 움직이는 겁니까?', PROMPT_TEMPLATE(spk_id=spk_id, emotion="", pitch="", tone="dynamic", infer=True), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[8]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2('어느 맑은 봄날, 바람에 이리저리 휫날리는 나뭇가지를 바라보며 제자가 물었다. 스승님, 저것은 나뭇가지가 움직이는 겁니까? 바람이 움직이는 겁니까?', PROMPT_TEMPLATE(spk_id=spk_id, emotion="", pitch="", tone="mono", infer=True), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[9]}", j['tts_speech'], cosyvoice.sample_rate)


#for i, j in enumerate(cosyvoice.inference_zero_shot('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다.', prompt_speech_16k, stream=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
