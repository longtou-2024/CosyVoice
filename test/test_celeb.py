import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.') # for cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pathlib import Path

celect_script = [
    {
        #"tag": "happy 3",
        "tag": "happy",
        "text": '와~ 드디어 우리 게임 출시됐어! 오늘은 밤새 파티다!',
    },
    {
        #"tag": "sad 2",
        "tag": "sad",
        "text": '그때 그렇게 말하지 말 걸... 아직도 마음에 남아.',
    },
    {
        "tag": "anxious",
        "text": '혹시 답장 아직 안 온 거예요? 뭔가 문제가 있는 건 아닌지...',
    },
    {
        #"tag": "angry 2",
        "tag": "angry",
        "text": '아니, 왜 자꾸 그 사람 얘기만 하는 거야?',
    },
    {
        "tag": "neutral",
        "text": '그날 이후, 둘은 다시 연락하지 않았다.',
    },
    {
        "tag": "embarrassed",
        "text": '아, 그 얘긴 그냥... 농담이었어요. 진짜예요...',
    },
    {
        #"tag": "sad 3",
        "tag": "sad",
        "text": '주변엔 늘 사람들이 있는데... 왜 이렇게 외로울까.',
    },
    {
        #"tag": "angry 3",
        "tag": "angry",
        "text": '그만 좀 해! 말했잖아, 난 그런 거 관심 없다고!',
    },
    {
        #"tag": "happy 1",
        "tag": "happy",
        "text": '비 온 뒤에 무지개가 떴다. 마음 한편이 따뜻해졌다.',
    },
    {
        "tag": "anxious",
        "text": '혹시... 내가 이상하게 보이진 않았죠? 다들 쳐다보는 것 같아서...',
    },
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

prompt_spk_speech_16k = dict()
for audio_path in Path("test/kast_emo").glob("*.wav"):
    uttid = audio_path.stem
    prompt_spk_speech_16k[uttid] = load_wav(audio_path, 16000)

prompt_spk_sent = dict()
for line in open("test/kast_emo/text", 'r').readlines():
    uttid, transcript = line.split(' ', maxsplit=1)
    prompt_spk_sent[uttid] = transcript



for d_idx, spk_id in enumerate(prompt_spk_sent):
    if spk_id != "pororo": continue
    prompt_sent = prompt_spk_sent[spk_id]
    prompt_speech_16k = prompt_spk_speech_16k[spk_id]
    for t_idx, item in enumerate(celect_script):
        tag = item["tag"]
        #tag = "독백체"
        this_text = item["text"]
        #for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag(tag, this_text), prefix_tag("neutral", prompt_sent), prompt_speech_16k, stream=False, text_frontend=False)):
        for i, j in enumerate(cosyvoice.inference_zero_shot(prefix_tag(tag, this_text), prompt_sent, prompt_speech_16k, stream=False, text_frontend=False)):
        #for i, j in enumerate(cosyvoice.inference_zero_shot(f" {this_text}", prompt_sent, prompt_speech_16k, stream=False, text_frontend=False)):
            torchaudio.save(f"{outdir}/{spk_id}_{t_idx}.wav", j['tts_speech'], cosyvoice.sample_rate)

    #if d_idx == 2:
    #    break

