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
azure_sent = "그녀, 결혼전만 해도 밝고 긍정적이던 여자, 그랬던 아내가 요즘 부쩍 걱정이 많아졌다."
lit_005_sent = "아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다."

#fout_names = ["base.wav", "fast.wav", "slow.wav"]
#fout_names = ["base.wav", "mono.wav", "dynamic.wav"]
#fout_names = ["base.wav", "low.wav", "high.wav"]
#fout_names = ["base.wav", "high.wav", "low.wav", "dynamic.wav", "mono.wav",
#              "happy.wav", "angry.wav", "sad.wav", "fast.wav", "slow.wav"]
fout_names = ["base.wav", "high.wav", "low.wav", "dynamic.wav", "mono.wav", "happy.wav", "angry.wav", "sad.wav", "fast.wav", "slow.wav", "whisper.wav", "laugh.wav"]

#for i, j in enumerate(cosyvoice.inference_zero_shot('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot('<high>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</high>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot('<low>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</low>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot('<dynamic>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</dynamic>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[3]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot('<mono>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</mono>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[4]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot('<기쁨>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</기쁨>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[5]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot('<분노>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</분노>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[6]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot('<슬픔>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</슬픔>', azure_sent, prompt_speech_16k, stream=False)):
#    torchaudio.save(f"{outdir}/{fout_names[7]}", j['tts_speech'], cosyvoice.sample_rate)


def pad_tag(text, tags):
    split = text.split(' ')
    tagged = []
    for word in split:
        tagged.append(tags[0] + word + tags[1])
    return " ".join(tagged)

this_text = '지금 상황에서 제일감은 역시 여기, 삼삼침투입니다.'
for i, j in enumerate(cosyvoice.inference_instruct2("<|azure|>" + this_text, "", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2("<|azure|>" + this_text, "여자 높은음", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2("<|azure|>" + this_text, "여자 낮은음", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2("<|azure|>" + this_text, "여자 다이나믹톤", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[3]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2("<|azure|>" + this_text, "여자 모노톤", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[4]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2("<|azure|>" + this_text, "기쁨", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[5]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2("<|azure|>" + this_text, "분노", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[6]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2("<|azure|>" + this_text, "슬픔", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[7]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2("<|azure|>" + pad_tag(this_text, ("<fast>", "</fast>")), "", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[8]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2("<|azure|>" + pad_tag(this_text, ("<slow>", "</slow>")), "", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[9]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2("<|azure|>" + this_text, "속삭임", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[10]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2("<|azure|>" + "지금 상황에서 [laughter] 제일감은 역시 여기, <laughter>삼삼침투입니다.</laughter>", "", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[11]}", j['tts_speech'], cosyvoice.sample_rate)

#fout_names = ["base.wav", "s1.wav", "s2.wav", "s3.wav", "s4.wav", "s5.wav", "s6.wav"]
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "중계체 분노 3", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "대화체 분노 3", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "낭독체 분노 3", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[3]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "친절체 분노 3", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[4]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "독백체 분노 3", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[5]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "구연체 분노 3", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[6]}", j['tts_speech'], cosyvoice.sample_rate)

#for i, j in enumerate(cosyvoice.inference_zero_shot(this_text, azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot("중계체 분노 3<|endofprompt|>" + this_text, azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot("대화체 분노 3<|endofprompt|>" + this_text, azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot("낭독체 분노 3<|endofprompt|>" + this_text, azure_sent,  prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[3]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot("친절체 분노 3<|endofprompt|>" + this_text, azure_sent,  prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[4]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot("독백체 분노 3<|endofprompt|>" + this_text, azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[5]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot("구연체 분노 3<|endofprompt|>" + this_text, azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[6]}", j['tts_speech'], cosyvoice.sample_rate)







#this_text = '안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.'
#for i, j in enumerate(cosyvoice.inference_zero_shot(this_text, lit_005_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot(pad_tag(this_text, ("<fast>", "</fast>")), lit_005_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot(pad_tag(this_text, ("<slow>", "</slow>")), lit_005_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
