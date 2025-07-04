import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.') # for cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pathlib import Path


outdir = "outdir"
Path(outdir).mkdir(exist_ok=True)
#fout_names = ["llm_base.wav", "llm_laugh.wav", "llm_whisper.wav"]
#
#cosyvoice = CosyVoice2('/home/longtou.2024/mount/longtou/saved/cosyvoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)
#
#prompt_speech_16k = load_wav('./test/ke_kim/F-A3-D-005-0051.wav', 16000)
#
#for i, j in enumerate(cosyvoice.inference_zero_shot('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다.', prompt_speech_16k, stream=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
#
#for i, j in enumerate(cosyvoice.inference_zero_shot('[laughter] 안녕하십니까, 오늘 오전 날씨는 맑고 <laughter>오후는</laughter> 구름이 조금 끼겠습니다. [laughter]', '아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다.', prompt_speech_16k, stream=False)):
#    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
#
#for i, j in enumerate(cosyvoice.inference_instruct2('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '속삭임', prompt_speech_16k, stream=False)):
#    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)

fout_names = ["llm_laugh1.wav", "llm_laugh2.wav", "llm_whisper.wav"]

cosyvoice = CosyVoice2('/home/longtou.2024/mount/longtou/saved/cosyvoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

prompt_speech_16k = load_wav('./test/ke_kim/F-A3-D-005-0051.wav', 16000)


for i, j in enumerate(cosyvoice.inference_zero_shot('<laughter>하하하</laughter> 그게 정말이야? [laughter] 너 진짜 웃기다! [laughter]', '아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다.', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_zero_shot('<laughter>히히히</laughter> 고작 그정도로는 나한테 안된다구 <laughter>푸하하하</laughter>', '아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다.', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct2('저기요, 죄송한데 이 물건 좀 치워주실 수 있을까요?', '속삭임', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
