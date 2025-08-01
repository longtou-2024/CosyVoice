import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.') # for cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pathlib import Path


shorts_script_tag = [
    ('mono high', '화살이 날아오고 있습니다, 왕녀님 조심하세요!'),
    ('hurry mono', '화살이 날아오고 있습니다, 왕녀님 조심하세요!'),
    ('hurry mono', '화살이 날아오고 있습니다, 왕녀님 조심하세요!'),
    ('hurry mono 3', '화살이 날아오고 있습니다, 왕녀님 조심하세요!'),
    ('hurry fear', '화살이 날아오고 있습니다, 왕녀님 조심하세요!'),
    ('hurry fear 3', '화살이 날아오고 있습니다, 왕녀님 조심하세요!'),
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


prompt_spk_speech_16k = load_wav("test/shorts/achird.wav", 16000)
#prompt_spk_speech_16k = load_wav('./test/ke_kim/F-A3-D-005-0051.wav', 16000)

prompt_spk_sent = dict()
for line in open("test/kast_emo/text", 'r').readlines():
    uttid, transcript = line.split(' ', maxsplit=1)
    prompt_spk_sent[uttid] = transcript
prompt_spk_sent = "긍지높던 황금의 왕국, 로이몬드"
#prompt_spk_sent = "아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다."

prompt_sent = prompt_spk_sent
prompt_speech_16k = prompt_spk_speech_16k


for t_idx, item in enumerate(shorts_script_tag):
    tag = item[0]
    this_text = item[1]
    #for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag(tag, wrap_tag("fast", this_text)), prompt_sent, prompt_speech_16k, stream=False, text_frontend=False)):
    for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag(tag, this_text), prompt_sent, prompt_speech_16k, stream=False, text_frontend=False)):
    #for i, j in enumerate(cosyvoice.inference_zero_shot(f" {this_text}", prompt_sent, prompt_speech_16k, stream=False, text_frontend=False)):
        torchaudio.save(f"{outdir}/{t_idx}.wav", j['tts_speech'], cosyvoice.sample_rate)
