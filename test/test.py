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
#fout_names = ["base.wav", "high.wav", "low.wav", "dynamic.wav", "mono.wav", "happy.wav", "angry.wav", "sad.wav", "fast.wav", "slow.wav", "whisper.wav", "laugh.wav"]
fout_names = ["fast.wav", "slow.wav"]

for i, j in enumerate(cosyvoice.inference_instruct2('<fast>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</fast>', "unkown 화자.", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2('<slow>안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.</slow>', "unkown 화자.", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)

#for i, j in enumerate(cosyvoice.inference_zero_shot('<fast>안녕하십니까</fast>, <fast>오늘</fast> <fast>오전</fast> <fast>날씨는</fast> <fast>맑고</fast> <fast>오후는</fast> <fast>구름이</fast> <fast>조금</fast> <fast>끼겠습니다.</fast>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot('<slow>안녕하십니까</slow>, <slow>오늘</slow> <slow>오전</slow> <slow>날씨는</slow> <slow>맑고</slow> <slow>오후는</slow> <slow>구름이</slow> <slow>조금</slow> <slow>끼겠습니다.</slow>', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', azure_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
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
spk = "azure"
chat_style = "낭독체"
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, f"{spk} 화자.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, f"{spk} 화자. 여자 높은음 스타일.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, f"{spk} 화자. 여자 낮은음 스타일.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, f"{spk} 화자. 여자 다이나믹톤 스타일.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[3]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, f"{spk} 화자. 여자 모노톤 스타일.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[4]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, f"{spk} 화자. {chat_style} 기쁨 3 스타일.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[5]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, f"{spk} 화자. {chat_style} 분노 3 스타일.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[6]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, f"{spk} 화자. {chat_style} 슬픔 3 스타일.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[7]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(pad_tag(this_text, ("<fast>", "</fast>")), f"{spk} 화자.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[8]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(pad_tag(this_text, ("<slow>", "</slow>")), f"{spk} 화자.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[9]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, f"{spk} 화자. 속삭임 스타일.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[10]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2("<laughter>푸하하하</laughter> 지금 상황에서 [laughter] 제일감은 역시 여기, <laughter>삼삼침투입니다.</laughter>", f"{spk} 화자.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[11]}", j['tts_speech'], cosyvoice.sample_rate)


this_text = '지금 상황에서 제일감은 역시 여기, 삼삼침투입니다.'
chat_style = "낭독체"
#fout_names = ["debug.wav"]
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "여자 높은음 스타일.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "여자 낮은음 스타일.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "여자 다이나믹톤 스타일.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[3]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "여자 모노톤 스타일.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[4]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, f"{chat_style} 기쁨 3 스타일.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[5]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, f"{chat_style} 분노 3 스타일.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[6]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, f"{chat_style} 슬픔 3 스타일.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[7]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(pad_tag(this_text, ("<fast>", "</fast>")), "unkown 화자.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[8]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(pad_tag(this_text, ("<slow>", "</slow>")), "unkown 화자.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[9]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "unkown 화자. 속삭임 스타일.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[10]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_instruct2("<laughter>푸하하하</laughter> 지금 상황에서 [laughter] 제일감은 역시 여기, <laughter>삼삼침투입니다.</laughter>", "unkown 화자.", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[11]}", j['tts_speech'], cosyvoice.sample_rate)
