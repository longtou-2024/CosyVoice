import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.') # for cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pathlib import Path

### 
# 중계체, 낭독체, 대화체, 애니체, 친절체, 독백체, 구연체
# happy, sad, angry
# embarrassed, neutral, anxious, hurt
###

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
fout_names = ["base.wav", "high.wav", "low.wav", "dynamic.wav", "mono.wav",
              "happy.wav", "angry.wav", "sad.wav", "fast.wav", "slow.wav", "whispering.wav",
              "sport.wav", "recite.wav", "chat.wav", "laugh.wav", "embarrassed.wav", "anxious.wav", "hurt.wav",
                "surprise.wav", "joy.wav", "doubt.wav", "fear.wav", "kind.wav", "hurry.wav", "serious.wav", "dry.wav", "shy.wav", "unpleasure.wav" ,"hesitate.wav", "tease.wav"
              ]

def pad_tag(text, tags):
    split = text.split(' ')
    tagged = []
    for word in split:
        tagged.append(tags[0] + word + tags[1])
    return " ".join(tagged)

#this_text = '지금 상황에서 제일감은 역시 여기, 삼삼침투입니다.'
this_text = '안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.'


for i, j in enumerate(cosyvoice.inference_zero_shot("<neutral>" + this_text + "</neutral>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[0]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<high>" + this_text + "</high>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[1]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<low>" + this_text + "</low>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[2]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<dynamic>" + this_text + "</dynamic>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[3]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<mono>" + this_text + "</mono>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[4]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<happy>" + this_text + "</happy>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[5]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<angry>" + this_text + "</angry>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[6]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<sad>" + this_text + "</sad>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[7]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<chat>" + pad_tag(this_text, ("<fast>", "</fast>")) + "</chat>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[8]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<chat>" + pad_tag(this_text, ("<slow>", "</slow>")) + "</chat>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[9]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "속삭임", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[10]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<sport>" + this_text + "</sport>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[11]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<recite>" + this_text + "</recite>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[12]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<chat>" + this_text + "</chat>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[13]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<chat>안녕하십니까, [laughter] 오늘 오전 날씨는 맑고 오후는 <laughter>구름이 조금 끼겠습니다.</laughter></chat>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[14]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<embarrassed>" + this_text + "</embarrassed>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[15]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<anxious>" + this_text + "</anxious>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[16]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<hurt>" + this_text + "</hurt>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[17]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<surprise>" + this_text + "</surprise>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[18]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<joy>" + this_text + "</joy>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[19]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<doubt>" + this_text + "</doubt>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[20]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<fear>" + this_text + "</fear>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[21]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<kind>" + this_text + "</kind>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[22]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<hurry>" + this_text + "</hurry>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[23]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<serious>" + this_text + "</serious>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[24]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<dry>" + this_text + "</dry>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[25]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<shy>" + this_text + "</shy>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[26]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<unpleasure>" + this_text + "</unpleasure>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[27]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<hesitate>" + this_text + "</hesitate>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[28]}", j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot("<tease>" + this_text + "</tease>", "<chat>" + azure_sent + "</chat>", prompt_speech_16k, stream=False, text_frontend=False)):
    torchaudio.save(f"{outdir}/{fout_names[29]}", j['tts_speech'], cosyvoice.sample_rate)




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
