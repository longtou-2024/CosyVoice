import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.') # for cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pathlib import Path

#from cosyvoice.dataset.webdataset import PROMPT_TEMPLATE

llm_path="/home/longtou.2024/projects/CosyVoice/examples/aihub/cosyvoice2/exp/cosyvoice2/llm/torch_ddp/llm_avg.pt"
cosyvoice = CosyVoice2('/home/longtou.2024/mount/longtou/saved/cosyvoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False, llm_path=llm_path)

outdir = "outdir"
Path(outdir).mkdir(exist_ok=True)
#prompt_speech_16k = load_wav('./test/azure/azure-01.wav', 16000)
prompt_speech_16k = load_wav('./test/ke_kim/F-A3-D-005-0051.wav', 16000)
#spk_id="azure"

fout_names = ["base.wav", "fast.wav", "slow.wav"]

for i, j in enumerate(cosyvoice.inference_zero_shot('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다.', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot('<fast>안녕하십니까</fast>, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다.', prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot('<slow>안녕하십니까</slow>, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다.', prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)




#for i, j in enumerate(cosyvoice.inference_instruct2('안녕하세요, 여러분의 친구, 박의주입니다.', PROMPT_TEMPLATE(spk_id=spk_id, emotion="", pitch="", tone="", infer=True), prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
