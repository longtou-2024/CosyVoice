import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.') # for cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice2LLM
from cosyvoice.utils.file_utils import load_wav
import torchaudio



# from model_dir, parse llm, flow, hift, + yaml
cosyvoice = CosyVoice2LLM(
    '/home/longtou.2024/mount/longtou/saved/cosyvoice/pretrained_models/CosyVoice2-0.5B', load_jit=False,
    yaml_path="/home/longtou.2024/projects/CosyVoice/examples/aihub/cosyvoice2/conf/cosyvoice2_lt2.yaml",
    load_trt=False, load_vllm=False, fp16=False)

prompt_speech_16k = load_wav('./test/ke_kim/F-A3-D-005-0051.wav', 16000)
tts_text_caption = ""
tts_text = "안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다."
prompt_text_caption = ""
prompt_text ="아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다."
for i, j in enumerate(cosyvoice.inference_zero_shot_caption(tts_text_caption, tts_text, prompt_text_caption, prompt_text, prompt_speech_16k, stream=False)):
#for i, j in enumerate(cosyvoice.inference_instruct2('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '제주도 방언', prompt_speech_16k, stream=False)):
#for i, j in enumerate(cosyvoice.inference_instruct2('요즘 시장 갈라면 사람이 많쿠게. 뭐사러 갔수다? 요즘은 물가가 많이 오르쿠게.', '제주도 방언', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
