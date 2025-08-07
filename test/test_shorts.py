import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.') # for cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pathlib import Path

shorts_script = [
    '긍지 높던 황금의 왕국, 로이몬드',
    '어느 날... 왕의 과욕으로 인해 찬란했던 영광은 사라지고 한순간에 무너져내렸는데...',
    '"잡아라!" 적들의 맹렬한 추격 속 죽기살기로 도망치고 있는 나, 제르이네',
    '나는 왕국의 여덟 번째 왕녀다',
    '언젠가 왕국을 다시 일으키려면, 후계자인 언니들이 꼭 살아남아야 하는데...',
    '이를 위해선, 내가 미끼가 되어야겠지...',
    '하지만 나는… 살고 싶어! 이렇게 허무하게 죽을 순 없어!',
    '그 순간... "왕녀님! 조심…!"',
    '나를 호위하던 기사는 눈앞에서 쓰러졌고',
    '날아온 화살은 내가 탄 말에 박히고 말았다',
    '흙바닥에 내동댕이쳐지며 의식이 흐려졌는데...',
    '얼마나 지났을까, 정신을 차려보니... 따스한 햇살과... 감옥?!',
    '이상한데? 무슨 감옥이 밖에 있어?!',
    '몸을 일으키려다 균형을 잃고 발을 헛디뎠는데...',
    '발밑에는 웬 막대기 하나가...?',
    '잠이 덜 깼나 싶어 눈을 비비려했더니',
    '내 손 대신 퍼덕이는... 깃털 달린 날개?!',
    '믿을 수 없어...! 내가… 내가 새라니?!',
]

shorts_script_tag = [
    ('neutral', '긍지 높던 황금의 왕국, 로이몬드'),
    ('serious', '어느 날... 왕의 과욕으로 인해 찬란했던 영광은 사라지고 한순간에 무너져내렸는데...'),
    ('hurry', '"잡아라!" 적들의 맹렬한 추격 속 죽기살기로 도망치고 있는 나, 제르이네'),
    ('serious', '나는 왕국의 여덟 번째 왕녀다'),
    ('anxious', '언젠가 왕국을 다시 일으키려면, 후계자인 언니들이 꼭 살아남아야 하는데...'),
    ('hurt', '이를 위해선, 내가 미끼가 되어야겠지...'),
    ('fear', '하지만 나는… 살고 싶어! 이렇게 허무하게 죽을 순 없어!'),
    ('surpirse', '그 순간... "왕녀님! 조심…!"'),
    ('hurt', '나를 호위하던 기사는 눈앞에서 쓰러졌고'),
    ('hurry', '날아온 화살은 내가 탄 말에 박히고 말았다'),
    ('hurt', '흙바닥에 내동댕이쳐지며 의식이 흐려졌는데...'),
    ('doubt', '얼마나 지났을까, 정신을 차려보니... 따스한 햇살과... 감옥?!'),
    ('embarrassed', '이상한데? 무슨 감옥이 밖에 있어?!'),
    ('hesitate', '몸을 일으키려다 균형을 잃고 발을 헛디뎠는데...'),
    ('doubt', '발밑에는 웬 막대기 하나가...?'),
    ('hesitate', '잠이 덜 깼나 싶어 눈을 비비려했더니'),
    ('surprise', '내 손 대신 퍼덕이는... 깃털 달린 날개?!'),
    ('surprise 3', '믿을 수 없어...! 내가… 내가 새라니?!'),
]

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


prompt_spk_speech_16k = load_wav("test/shorts/short_sample.wav", 16000)

prompt_spk_sent = dict()
for line in open("test/kast_emo/text", 'r').readlines():
    uttid, transcript = line.split(' ', maxsplit=1)
    prompt_spk_sent[uttid] = transcript
prompt_spk_sent = "긍지높던 황금의 왕국, 로이몬드"

prompt_sent = prompt_spk_sent
prompt_speech_16k = prompt_spk_speech_16k
#for t_idx, item in enumerate(shorts_script):
#    this_text = item
#    for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag("neutral", this_text), prompt_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#    #for i, j in enumerate(cosyvoice.inference_zero_shot(f" {this_text}", prompt_sent, prompt_speech_16k, stream=False, text_frontend=False)):
#        torchaudio.save(f"{outdir}/{t_idx}.wav", j['tts_speech'], cosyvoice.sample_rate)


for t_idx, item in enumerate(shorts_script_tag):
    tag = item[0]
    this_text = item[1]
    for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag(tag, this_text), prompt_sent, prompt_speech_16k, stream=False, text_frontend=False)):
    #for i, j in enumerate(cosyvoice.inference_zero_shot(f" {this_text}", prompt_sent, prompt_speech_16k, stream=False, text_frontend=False)):
        torchaudio.save(f"{outdir}/{t_idx}.wav", j['tts_speech'], cosyvoice.sample_rate)
