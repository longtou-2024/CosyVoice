import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.') # for cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pathlib import Path

### 
# 중계체, 낭독체, 대화체, 애니체, 친절체, 독백체, 구연체
# happy 3, sad 3, angry 3
# 
# embarrassed, neutral, anxious, hurt
# surprise, joy, doubt, fear, kind, hurry, serious, dry, shy, unpleasure, hesitate, tease
# chat teen, chat adult, laugh
# 속삭임, fast, slow, high, low, dynamic, mono
###

def prefix_tag(tag, text):
    return f"<|tag_start|>{tag}<|tag_end|>{text}"

def wrap_tag(tag, text):
    split = text.split(' ')
    tagged = []
    for word in split:
        tagged.append(f"<{tag}>{word}</{tag}>")
    return " ".join(tagged)

llm_path="/home/longtou.2024/projects/CosyVoice/examples/aihub/cosyvoice2/exp/cosyvoice2/llm/torch_ddp/llm_avg.pt"
cosyvoice = CosyVoice2('/home/longtou.2024/mount/longtou/saved/cosyvoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False, llm_path=llm_path)

outdir = "outdir"
Path(outdir).mkdir(exist_ok=True)
#prompt_speech_16k = load_wav('./test/azure/azure-01.wav', 16000)
prompt_speech_16k = load_wav('./test/ke_kim/F-A3-D-005-0051.wav', 16000)
#spk_id="azure"
azure_sent = "그녀, 결혼전만 해도 밝고 긍정적이던 여자, 그랬던 아내가 요즘 부쩍 걱정이 많아졌다."
lit_005_sent = "아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다."
prompt_sent = lit_005_sent
prompt_tag = "neutral"

fout_names = ["base.wav", "fast.wav", "slow.wav", "high.wav", "low.wav", "dynamic.wav", "mono.wav",
              "whispering.wav", "laugh.wav", "chat_teen.wav", "chat_adult.wav", "sport.wav", "recite.wav", "conversation.wav", "anime.wav", "polite.wav", "sole.wav", "theme.wav",
              "happy.wav", "sad.wav", "angry.wav", "embarrassed.wav", "anxious.wav", "hurt.wav", "surprise.wav", "joy.wav", "doubt.wav", "fear.wav", "kind.wav", "hurry.wav", "serious.wav", "dry.wav", "shy.wav", "unpleasure.wav", "hesitate.wav", "tease.wav"
              ]


#this_text = '지금 상황에서 제일감은 역시 여기, 삼삼침투입니다.'
this_text = '안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.'


#fout_names = ["base.wav", "fast.wav", "slow.wav", "high.wav", "low.wav"]
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("neutral", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("neutral", wrap_tag("fast", this_text)), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("neutral", wrap_tag("slow", this_text)), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("high", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[3]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("low", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[4]}", j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("dynamic", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[5]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("mono", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[6]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "속삭임", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[7]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("안녕하십니까, [laughter] 오늘 오전 날씨는 맑고 <laughter>오후는 구름이 조금 끼겠습니다.</laughter>", prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[8]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("chat teen", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[9]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("chat adult", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[10]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("중계체", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[11]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("낭독체", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[12]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("대화체", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[13]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("애니체", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[14]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("친절체", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[15]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("독백체", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[16]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("구연체", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[17]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("happy 3", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[18]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("sad 3", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[19]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("angry 3", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[20]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("embarrassed", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[21]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("anxious", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[22]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("hurt", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[23]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("surprise", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[24]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("joy", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[25]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("doubt", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[26]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("fear", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[27]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("kind", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[28]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("hurry", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[29]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("serious", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[30]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("dry", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[31]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("shy", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[32]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("unpleasure", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[33]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("hesitate", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[34]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("tease", this_text), prefix_tag(prompt_tag, prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[35]}", j['tts_speech'], cosyvoice.sample_rate)




# 1) 웃음 테스트
#this_text = "정말 그 말 하는 순간 너의 얼굴은, 하하하, 완전 토끼 같았어!"
#fout_names = ["base.wav", "laugh1.wav", "laugh2.wav", "laugh3.wav", "laugh4.wav"]
# 2) 속도를 이용한 강조 테스트
#this_text = "빠르게 말할때는 빠르게 말하고, 천천히 말해야 할때는 천천히 말하는게 좋습니다."
#this_text = "상처를 치료해줄 사람 어디 없나, 가만히 놔 두다간 끊임없이 덧나, 사랑도 사람도 너무나도 겁나, 혼자인게 무서워 난 잊혀질까 두려워"
#fout_names = ["base.wav", "fast.wav",]
# 3) 스타일 테스트
#this_text = "나 진짜 그런 의도가 있었던 건 아니었어, 정말이야"
#fout_names = ["base.wav", "chat.wav", "happy.wav", "angry.wav" ,"sad.wav"]
# 4) 스타일2 테스트
#this_text = "자, 이제는 바론 앞에서 치열한 심리전! 양 팀 스펠 다 돌았어요, 이 한타가 사실상 오늘 경기의 운명을 가를 겁니다!"
#fout_names = ["base.wav", "sport.wav", "recite.wav", "f_mono.wav" ,"f_dynamic.wav", "f_low.wav", "f_high.wav"]
#for i, j in enumerate(cosyvoice.inference_zero_shot("<neutral>" + this_text + "</neutral>", "<neutral>" + azure_sent + "</neutral>", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot("<sport>" + this_text + "</sport>", "<neutral>" + azure_sent + "</neutral>", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot("<recite>" + this_text + "</recite>", "<neutral>" + azure_sent + "</neutral>", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot("<f_mono>" + this_text + "</f_mono>", "<neutral>" + azure_sent + "</neutral>", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[3]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot("<f_dynamic>" + this_text + "</f_dynamic>", "<neutral>" + azure_sent + "</neutral>", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[4]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot("<f_low>" + this_text + "</f_low>", "<neutral>" + azure_sent + "</neutral>", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[5]}", j['tts_speech'], cosyvoice.sample_rate)
#for i, j in enumerate(cosyvoice.inference_zero_shot("<f_high>" + this_text + "</f_high>", "<neutral>" + azure_sent + "</neutral>", prompt_speech_16k, stream=False, text_frontend=False)):
#    torchaudio.save(f"{outdir}/{fout_names[6]}", j['tts_speech'], cosyvoice.sample_rate)
