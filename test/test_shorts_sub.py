import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.') # for cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pathlib import Path

# ('low', '우연이라기엔 지나치게 이상하다.'),
#('low', '어떻게 훈련도 받지 못한 새가 기밀문서를 들고 정확히 나에게 온 거지?'),
    #('low 3', '수색해, 확실히, 정상이 아닌 것 같긴 하군'),
    #('serious', '이걸 찾나?'),
    #('mono', '집착하지 마라'),
    #('low 1', '...이 녀석을 보내려고 한다'),
    #('serious 1', '그런 일이 있다'),

    #('dynamic 3', '저 새는 미쳤습니다. 확실히!'),
    #('hurry 3', '무슨 전갈을 보내려고 하십니까'),
    #('serious', '전령새 대부분이 돌아오지 않았다, 이제 남은 전령새는 이 새 하나다'),
shorts_script_tag = [
    ('dry', '전령새 대부분이 돌아오지 않았다, 이제 남은 전령새는 이 새 하나다'),
    ('serious', '전령새 대부분이 돌아오지 않았다, 이제 남은 전령새는 이 새 하나다'),
    ('low', '전령새 대부분이 돌아오지 않았다, 이제 남은 전령새는 이 새 하나다'),
    ('fear', '전령새 대부분이 돌아오지 않았다, 이제 남은 전령새는 이 새 하나다'),
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


prompt_spk_speech_16k = load_wav('./test/shorts2/prince.wav', 16000)
#prompt_spk_speech_16k = load_wav("test/shorts2/soldier.wav", 16000)

prompt_spk_sent = "짐승도 암살에 쓰나?"
#prompt_spk_sent = "애가 왜이러지? 아픈가? 보고를 올려야하나?"

prompt_sent = prompt_spk_sent
prompt_speech_16k = prompt_spk_speech_16k


for t_idx, item in enumerate(shorts_script_tag):
    tag = item[0]
    this_text = item[1]

    if tag == "":
        tts_text = f" {this_text}"
    else:
        tts_text = prefix_tag(tag, this_text)

    for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, prompt_sent, prompt_speech_16k, stream=False, text_frontend=False)):
        torchaudio.save(f"{outdir}/{t_idx}.wav", j['tts_speech'], cosyvoice.sample_rate)
