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
spk_id = "azure"
prompt_tag = "neutral"

prompt_spk_speech_16k = {
    "azure": load_wav('./test/azure/azure-01.wav', 16000),
    "kim": load_wav('./test/ke_kim/F-A3-D-005-0051.wav', 16000),
}
prompt_speech_16k = prompt_spk_speech_16k[spk_id]

prompt_spk_sent = {
    "azure":  "그녀, 결혼전만 해도 밝고 긍정적이던 여자, 그랬던 아내가 요즘 부쩍 걱정이 많아졌다.",
    "kim": "아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다.",
                   }
prompt_sent = prompt_spk_sent[spk_id]


# Test various style tag
#
# 1) 웃음소리; laugh
# 2) 속삭임
# 3) 느리게 강조하며; slow
# 4) 빠르고 급하게; fast, hurry
# 5) 대화체; chat teen, chat adult, 중계체, 낭독체, 대화체, 애니체, 친절체, 독백체, 구연체
# 6) 톤변화
# 6.1) 톤 높낮이; high, low
# 6.2) 톤 다이나믹; mono, dynamic
# 7) 감정과 강도; happy, sad, angry
# 8) 기타 감정; embarrassed, anxious, hurt, surprise, joy, doubt, fear, kind, serious, dry, shy, unpleasure, hesitate, tease



# 1) 웃음 테스트
tts_text = [
    '<laughter>킥킥</laughter>, 너만 그런 거 아니야. 나도 깔깔 웃었어.',
    '<laughter>푸하하하</laughter>, 진짜 그런 상황이면 나라도 웃었을걸?',
    '<laughter>크크크</laughter>, 시도해봐. 내가 더 잘 웃길지도 몰라.',
    '<laughter>하하하</laughter>, 나도 내 심장이 어떻게 뛰었는지 모르겠어.',
    '[laughter] 나도 내 심장이 어떻게 뛰었는지 모르겠어.',
]
for spk_id in ("azure", "kim"):
    prompt_speech_16k = prompt_spk_speech_16k[spk_id]
    prompt_sent = prompt_spk_sent[spk_id]
    for t_idx, this_text in enumerate(tts_text):
        for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("chat adult", this_text), prefix_tag("neutral", prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
            torchaudio.save(f"{outdir}/{spk_id}_laugh_{t_idx}.wav", j['tts_speech'], cosyvoice.sample_rate)


# 2) 속삭임 테스트
#tts_text = [
#    '조용히 해... 들킬지도 몰라.',
#    '이건 우리만의 비밀이야. 아무에게도 안 들키게.',
#]
#for spk_id in ("azure", "kim"):
#    prompt_speech_16k = prompt_spk_speech_16k[spk_id]
#    prompt_sent = prompt_spk_sent[spk_id]
#    for t_idx, this_text in enumerate(tts_text):
#        for i, j in enumerate(cosyvoice.inference_instruct2(this_text, "속삭임", prompt_speech_16k, stream=False, text_frontend=False)):
#            torchaudio.save(f"{outdir}/{spk_id}_whisper_{t_idx}.wav", j['tts_speech'], cosyvoice.sample_rate)


# 3) 느리게 강조 테스트
#tts_text = [
#    '중요한 안전 수칙을 꼭 기억하세요.',
#    '이 문서는 기밀 정보이므로 외부 유출 금지입니다.',
#]
#for spk_id in ("azure", "kim"):
#    prompt_speech_16k = prompt_spk_speech_16k[spk_id]
#    prompt_sent = prompt_spk_sent[spk_id]
#    for t_idx, this_text in enumerate(tts_text):
#        for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("대화체", wrap_tag("slow", this_text)), prefix_tag("neutral", prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
#            torchaudio.save(f"{outdir}/{spk_id}_slow_{t_idx}.wav", j['tts_speech'], cosyvoice.sample_rate)
#        for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("대화체", this_text), prefix_tag("neutral", prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
#            torchaudio.save(f"{outdir}/{spk_id}_slow_base_{t_idx}.wav", j['tts_speech'], cosyvoice.sample_rate)

# 4) 빠르고 급하게 테스트
#tts_text = [
#    '빨리, 지금 당장 움직여야 해!',
#    '바로 이쪽으로 와, 설명할 시간이 없어!',
#    '멈춰! 지금 당장 멈춰야 해!'
#]
#for spk_id in ("azure", "kim"):
#    prompt_speech_16k = prompt_spk_speech_16k[spk_id]
#    prompt_sent = prompt_spk_sent[spk_id]
#    for t_idx, this_text in enumerate(tts_text):
#        for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("chat adult", wrap_tag("fast", this_text)), prefix_tag("neutral", prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
#            torchaudio.save(f"{outdir}/{spk_id}_fast_{t_idx}.wav", j['tts_speech'], cosyvoice.sample_rate)
#        for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("chat adult", this_text), prefix_tag("neutral", prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
#            torchaudio.save(f"{outdir}/{spk_id}_fast_base_{t_idx}.wav", j['tts_speech'], cosyvoice.sample_rate)
#        for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("hurry", this_text), prefix_tag("neutral", prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
#            torchaudio.save(f"{outdir}/{spk_id}_hurry_{t_idx}.wav", j['tts_speech'], cosyvoice.sample_rate)

# 5) 대화체 테스트
#tts_text = [
#    ('독백체', '왜 나는 매번 이렇게 망설이는 걸까… 하하, 이럴 때일수록 담대해야 하는데, 마음은 자꾸만 작아지기만 해.'),
#    ('대화체', '왜 나는 매번 이렇게 망설이는 걸까… 하하, 이럴 때일수록 담대해야 하는데, 마음은 자꾸만 작아지기만 해.'),
#    ('chat teen', '왜 나는 매번 이렇게 망설이는 걸까… 하하, 이럴 때일수록 담대해야 하는데, 마음은 자꾸만 작아지기만 해.'),
#    ('chat adult', '왜 나는 매번 이렇게 망설이는 걸까… 하하, 이럴 때일수록 담대해야 하는데, 마음은 자꾸만 작아지기만 해.'),
#    ('구연체', '어느 날, 작은 골목길에서 한 남자가 말했어. "여기가 바로 우리 시작점이야." 그리고 모두가 그 말에 힘을 얻었단다.'),
#    ('낭독체', '어느 날, 작은 골목길에서 한 남자가 말했어. "여기가 바로 우리 시작점이야." 그리고 모두가 그 말에 힘을 얻었단다.'),
#    ('중계체', '마침내 오리아나 오인궁 적중! 글로벌 골드는 역전되었습니다!'),
#]
#for spk_id in ("azure", "kim"):
#    prompt_speech_16k = prompt_spk_speech_16k[spk_id]
#    prompt_sent = prompt_spk_sent[spk_id]
#    for t_idx, x in enumerate(tts_text):
#        tag = x[0]
#        this_text = x[1]
#        for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag(tag, this_text), prefix_tag("neutral", prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
#            torchaudio.save(f"{outdir}/{spk_id}_{tag}.wav", j['tts_speech'], cosyvoice.sample_rate)


# 6) 톤 변화 테스트
#tts_text = [
#    ('dynamic', '이 순간이 바로 우리가 오랫동안 기다려온 기회야. 절대 놓치지 말고 힘을 다해 잡아야 해.'),
#    ('mono', '이 순간이 바로 우리가 오랫동안 기다려온 기회야. 절대 놓치지 말고 힘을 다해 잡아야 해.'),
#    ('high', '조용히 해봐, 분명 무언가 이상한 소리가 들렸어… 긴장하지 말고 귀 기울여야 해.'),
#    ('low', '조용히 해봐, 분명 무언가 이상한 소리가 들렸어… 긴장하지 말고 귀 기울여야 해.'),
#]
#for spk_id in ("azure", "kim"):
#    prompt_speech_16k = prompt_spk_speech_16k[spk_id]
#    prompt_sent = prompt_spk_sent[spk_id]
#    for t_idx, x in enumerate(tts_text):
#        tag = x[0]
#        this_text = x[1]
#        for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag(tag, this_text), prefix_tag("neutral", prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
#            torchaudio.save(f"{outdir}/{spk_id}_{tag}.wav", j['tts_speech'], cosyvoice.sample_rate)


# 7) 감정 강도 테스트
#tts_text = [
#    ('happy', '드디어 모든 준비가 끝났어! 오늘은 정말 최고의 하루가 될 것 같아.'),
#    ('happy 1', '드디어 모든 준비가 끝났어! 오늘은 정말 최고의 하루가 될 것 같아.'),
#    ('happy 2', '드디어 모든 준비가 끝났어! 오늘은 정말 최고의 하루가 될 것 같아.'),
#    ('happy 3', '드디어 모든 준비가 끝났어! 오늘은 정말 최고의 하루가 될 것 같아.'),
#    ('sad', '그 사람이 떠난 후로 매일이 조금씩 어두워지는 것 같아.'),
#    ('sad 1', '그 사람이 떠난 후로 매일이 조금씩 어두워지는 것 같아.'),
#    ('sad 2', '그 사람이 떠난 후로 매일이 조금씩 어두워지는 것 같아.'),
#    ('sad 3', '그 사람이 떠난 후로 매일이 조금씩 어두워지는 것 같아.'),
#    ('angry', '내가 이렇게까지 분노하는 이유를 알게 될 거야, 분명히!'),
#    ('angry 1', '내가 이렇게까지 분노하는 이유를 알게 될 거야, 분명히!'),
#    ('angry 2', '내가 이렇게까지 분노하는 이유를 알게 될 거야, 분명히!'),
#    ('angry 3', '내가 이렇게까지 분노하는 이유를 알게 될 거야, 분명히!'),
#]
#for spk_id in ("azure", "kim"):
#    prompt_speech_16k = prompt_spk_speech_16k[spk_id]
#    prompt_sent = prompt_spk_sent[spk_id]
#    for t_idx, x in enumerate(tts_text):
#        tag = x[0]
#        this_text = x[1]
#        for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag(tag, this_text), prefix_tag("neutral", prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
#            torchaudio.save(f"{outdir}/{spk_id}_{tag.replace(' ', '_')}.wav", j['tts_speech'], cosyvoice.sample_rate)



# 8) 기타 감정 테스트
#tts_text = [
#    ('happy', '오늘은 날씨도 좋고, 마음까지 맑아서 정말 기분이 좋아!'),
#    ('sad', '그 말을 듣고 나니, 가슴 한켠이 너무 아파서 견딜 수가 없어.'),
#    ('angry', '도대체 무슨 생각으로 그런 짓을 한 거야? 정말 참을 수 없어!'),
#    ('embarrassed', '아, 내가 왜 그런 말을 했지?... 너무 얼굴이 뜨거워져서 이 순간을 빨리 지나가고 싶어.'),
#    ('anxious', '시간이 점점 다가올수록 가슴이 쿵쾅거리고, 떨리는 걸 숨길 수가 없어.'),
#    ('hurt', '네 말 한마디가 이렇게 깊이 다칠 줄은 몰랐어.'),
#    ('surprise', '뭐라고? 그럴 줄은 전혀 예상 못했어, 정말 깜짝 놀랐어!'),
#    ('joy', '이 순간을 함께할 수 있어서 정말 행복하고, 마음이 벅차올라!'),
#    ('doubt', '정말 그 사람이 한 말이 사실인지, 솔직히 믿기 힘들어.'),
#    ('fear', '어둠 속에서 누군가 다가오는 소리가 들리자, 순간 몸이 얼어붙었어.'),
#    ('kind', '걱정 마, 내가 언제나 네 옆에 있을게. 같이 이겨내 보자.'),
#    ('hurry', '시간 없어, 빨리 움직여야 해! 늦으면 안 돼!'),
#    ('serious', '이 문제는 가볍게 넘어갈 수 없어. 우리 모두 진지하게 생각해야 해.'),
#    ('dry', '음, 그렇게 하는 게 맞겠지. 그게 다야.'),
#    ('shy', '그… 그 말, 고마워. 나, 조금 쑥스러워서… 어색하네.'),
#    ('unpleasure', '솔직히 말하면, 그 상황이 너무 불편하고 마음에 들지 않았어'),
#    ('hesitate', '음… 해야 할지 말아야 할지 잘 모르겠어. 조금만 더 생각해볼게.'),
#    ('tease', '하하, 그런 표정 지으니까 더 웃기네! 좀 웃어봐, 너무 딱딱해.'),
#]
#for spk_id in ("azure", "kim"):
#    prompt_speech_16k = prompt_spk_speech_16k[spk_id]
#    prompt_sent = prompt_spk_sent[spk_id]
#    for t_idx, x in enumerate(tts_text):
#        tag = x[0]
#        this_text = x[1]
#        for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag(tag, this_text), prefix_tag("neutral", prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
#            torchaudio.save(f"{outdir}/{spk_id}_{tag}.wav", j['tts_speech'], cosyvoice.sample_rate)
