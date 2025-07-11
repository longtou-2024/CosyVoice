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
azure_sent = "그녀, 결혼전만 해도 밝고 긍정적이던 여자, 그랬던 아내가 요즘 부쩍 걱정이 많아졌다."
lit_005_sent = "아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다."

fout_names = ["base.wav", "fast.wav", "slow.wav"]
#fout_names = ["base.wav", "mono.wav", "dynamic.wav"]
#fout_names = ["base.wav", "low.wav", "high.wav"]

#for i, j in enumerate(cosyvoice.inference_zero_shot('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot('<fast>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</fast>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot('<slow>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</slow>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)

#for i, j in enumerate(cosyvoice.inference_zero_shot('네? 네 네 말씀하세요', azure_sent, prompt_speech_16k, stream=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot('네? <fast>네</fast> <fast>네</fast> 말씀하세요', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot('네? <slow>네</slow> <slow>네</slow> 말씀하세요', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)

#for i, j in enumerate(cosyvoice.inference_zero_shot('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', azure_sent, prompt_speech_16k, stream=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot('<low>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</low>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot('<high>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</high>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)



#for i, j in enumerate(cosyvoice.inference_instruct2('지금 상황에서 제일감은 역시 여기, 삼삼침투입니다.', "", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2('지금 상황에서 <fast>제일감은</fast> 역시 <fast>여기</fast>, 삼삼침투입니다.', "", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2('지금 상황에서 <slow>제일감은</slow> 역시 <slow>여기</slow>, 삼삼침투입니다.', "", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)

def pad_tag(text, tags):
    split = text.split(' ')
    tagged = []
    for word in split:
        tagged.append(tags[0] + word + tags[1])
    return " ".join(tagged)

this_text = '안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.'
for i, j in enumerate(cosyvoice.inference_zero_shot(this_text, lit_005_sent, prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(pad_tag(this_text, ("<fast>", "</fast>")), lit_005_sent, prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(pad_tag(this_text, ("<slow>", "</slow>")), lit_005_sent, prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
