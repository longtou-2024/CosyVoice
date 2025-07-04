import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.') # for cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pathlib import Path


llm_path="/home/longtou.2024/projects/CosyVoice/examples/aihub/cosyvoice2/exp/cosyvoice2/llm/torch_ddp/llm_avg.pt"
cosyvoice = CosyVoice2('/home/longtou.2024/mount/longtou/saved/cosyvoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False, llm_path=llm_path)

outdir = "outdir"
Path(outdir).mkdir(exist_ok=True)
prompt_speech_16k = load_wav('./test/ke_kim/F-A3-D-005-0051.wav', 16000)

fout_names = ["base.wav", "laugh.wav", "laugh2.wav", "laugh3.wav", "whisper.wav",
              "high.wav", "low.wav", "monotone.wav", "dynamictone.wav", "happy.wav", "sad.wav", "angry.wav", "no_emotion.wav"]


for i, j in enumerate(cosyvoice.inference_zero_shot('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다.', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_zero_shot('안녕하십니까, [laughter] 오늘 오전 날씨는 맑고 <laughter>오후는</laughter> 구름이 조금 끼겠습니다. [laughter]', '아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다.', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_zero_shot('<laughter>하하하</laughter> 그게 정말이야? [laughter] 너 진짜 웃기다! [laughter]', '아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다.', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_zero_shot('<laughter>히히히</laughter> 고작 그정도로는 나한테 안된다구 <laughter>푸하하하</laughter>', '아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다.', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[3]}", j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct2('저기요, 죄송한데 이 물건 좀 치워주실 수 있을까요?', '속삭임', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[4]}", j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct2('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '남자 높은음', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[5]}", j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct2('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '남자 낮은음', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[6]}", j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct2('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '남자 모노톤', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[7]}", j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct2('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '남자 다이나믹톤', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[8]}", j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct2('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '기쁨', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[9]}", j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct2('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '슬픔', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[10]}", j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct2('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '분노', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[11]}", j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_instruct2('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '무감정', prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[12]}", j['tts_speech'], cosyvoice.sample_rate)
