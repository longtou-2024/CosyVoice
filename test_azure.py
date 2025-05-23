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


text = "희진아, 이번 주말에 시간 어때? 오, 진짜? 그럼 일요일에 우리 영화 볼래? 나도 그거 궁금했는데 ㅋㅋ 완전 딱이다 오후 2시 어때? 완전 좋지! 그럼 일요일 2시, 영화관에서 만나자!"

tts_text = text
tts_speeches = []
tts_text_token, tts_text_token_len = cosyvoice.frontend._extract_text_token("azure<|endofprompt|>" + tts_text)
model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
               'llm_embedding': spk_embedding, 'flow_embedding': spk_embedding}
for model_output in cosyvoice.model.tts(**model_input):
    tts_speeches.append(model_output['tts_speech'])
tts_speeches = torch.concat(tts_speeches, dim=1) # [C,T]
torchaudio.save(outdir / f'azure.wav', tts_speeches, cosyvoice.sample_rate)

