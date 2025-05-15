## SDK模型下载
#from modelscope import snapshot_download
#snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')

import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pathlib import Path
import torch

azure_spk_emb_path = "/home/longtou.2024/projects/CosyVoice/examples/azure/cosyvoice2/data/tr_no_dev/spk2embedding.pt"
spk_emb = torch.load(azure_spk_emb_path, map_location="cuda")
spk_embedding = torch.tensor(spk_emb["azure"]).to("cuda").unsqueeze(0) # (1,192)
#print(spk_embedding)
#import sys; sys.exit()


outdir = Path("outdir")
outdir.mkdir(parents=True, exist_ok=True)

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)

text = {
    "annoying": "싫어! 나 밥 안먹을꺼야 놀게 해줘~",
    "fairy": "고래는 멋있어요. 거대한 몸으로 바다를 거니는 모습, 상상만해도 설레죠.",
    "history": "임금이 태평한 태평성대를 보았느냐? 내 마음이 지옥이기에 그나마 세상이 평온한 것이다.",
    "joy": "많이 기다리셨죠? 드디어 최종 워크샵 장소가 결정되었습니다!",
    "ridicule": "너가 지금 나를 잡아먹으려고 하는구나?",
    "sad": "아이고 또 떨어졌니? 걱정이 늘었구나.",
    "sport": "경기가 종료됐습니다! 오늘의 승리는 티원이 거두었고, 최종 스코어는 이대영 입니다. 오늘의 엠브이피는 페이커 선수으로 선정됐습니다.",
}

for idx, prosody in enumerate(text):
    tts_text = text[prosody]
    tts_speeches = []
    tts_text_token, tts_text_token_len = cosyvoice.frontend._extract_text_token("azure<|endofprompt|>" + tts_text)
    model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                   'llm_embedding': spk_embedding, 'flow_embedding': spk_embedding}
    for model_output in cosyvoice.model.tts(**model_input):
        tts_speeches.append(model_output['tts_speech'])
    tts_speeches = torch.concat(tts_speeches, dim=1) # [C,T]
    torchaudio.save(outdir / f'azure_{prosody}.wav', tts_speeches, cosyvoice.sample_rate)


# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
#for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#    torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
