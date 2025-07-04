import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.') # for cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio



cosyvoice = CosyVoice2('/home/longtou.2024/mount/longtou/saved/cosyvoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

prompt_speech_16k = load_wav('./test/ke_kim/F-A3-D-005-0051.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('[laughter] 안녕하십니까, 오늘 오전 날씨는 맑고 <laughter>오후는</laughter> 구름이 조금 끼겠습니다. [laughter]', '아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다.', prompt_speech_16k, stream=False)):
#for i, j in enumerate(cosyvoice.inference_instruct2('안녕하십니까, 오늘 오전 날씨는 맑고 오후는 구름이 조금 끼겠습니다.', '속삭임', prompt_speech_16k, stream=False)):
#for i, j in enumerate(cosyvoice.inference_instruct2('요즘 시장 갈라면 사람이 많쿠게. 뭐사러 갔수다? 요즘은 물가가 많이 오르쿠게.', '제주도 방언', prompt_speech_16k, stream=False)):
#for i, j in enumerate(cosyvoice.inference_instruct2('마! 니 동수햄 아나? 나랑 지금 한따까리 할래?', '경상도 방언', prompt_speech_16k, stream=False)):
#for i, j in enumerate(cosyvoice.inference_zero_shot('마! 니 동수햄 아나? 나랑 지금 한따까리 할래?', '아빠 말씀에 엄마가 막내이모를 향해 화를 버럭 냈습니다.', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
