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
prompt_speech_16k = load_wav('./test/azure/azure-01.wav', 16000)
#prompt_speech_16k = load_wav('./test/ke_kim/F-A3-D-005-0051.wav', 16000)
#spk_id="azure"
azure_sent = "그녀, 결혼전만 해도 밝고 긍정적이던 여자, 그랬던 아내가 요즘 부쩍 걱정이 많아졌다."
lit_005_sent = "아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다."

#fout_names = ["base.wav", "fast.wav", "slow.wav"]
#fout_names = ["base.wav", "mono.wav", "dynamic.wav"]
#fout_names = ["base.wav", "low.wav", "high.wav"]
#fout_names = ["base.wav", "high.wav", "low.wav", "dynamic.wav", "mono.wav",
#              "happy.wav", "angry.wav", "sad.wav", "fast.wav", "slow.wav"]
fout_names = ["base.wav", "high.wav", "low.wav", "dynamic.wav", "mono.wav", "happy.wav", "angry.wav", "sad.wav"]

for i, j in enumerate(cosyvoice.inference_zero_shot('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot('<high>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</high>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot('<low>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</low>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot('<dynamic>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</dynamic>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[3]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot('<mono>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</mono>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[4]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot('<기쁨>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</기쁨>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[5]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot('<분노>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</분노>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[6]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot('<슬픔>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</슬픔>', azure_sent, prompt_speech_16k, stream=False)):
    torchaudio.save(f"{outdir}/{fout_names[7]}", j['tts_speech'], cosyvoice.sample_rate)


def pad_tag(text, tags):
    split = text.split(' ')
    tagged = []
    for word in split:
        tagged.append(tags[0] + word + tags[1])
    return " ".join(tagged)

this_text = '지금 상황에서 제일감은 역시 여기, 삼삼침투입니다.'
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "azure", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2("<high>" + this_text + "</high>", "azure", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2("<low>" + this_text + "</low>", "azure", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2("<dynamic>" + this_text + "</dynamic>", "azure", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[3]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2("<mono>" + this_text + "</mono>", "azure", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[4]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2("<기쁨>" + this_text + "</기쁨>", "azure", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[5]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2("<분노>" + this_text + "</분노>", "azure", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[6]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2("<슬픔>" + this_text + "</슬픔>", "azure", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[7]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(pad_tag(this_text, ("<fast>", "</fast>")), "azure", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[8]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(pad_tag(this_text, ("<slow>", "</slow>")), "azure", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[9]}", j['tts_speech'], cosyvoice.sample_rate)


#this_text = '안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.'
#for i, j in enumerate(cosyvoice.inference_zero_shot(this_text, lit_005_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot(pad_tag(this_text, ("<fast>", "</fast>")), lit_005_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot(pad_tag(this_text, ("<slow>", "</slow>")), lit_005_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
