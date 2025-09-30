import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('.') # for cosyvoice
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pathlib import Path
import json, io
import torch


spk_roles = {
    "m_narrator": ["N0217-01-21-00", "5042_G1A2E7_KJB_004142", "5039_G1A2E7_KIM_000872", "1491_G1A2E7_JJW_001089"],
    "f_narrator": ["9035_G2A1E7_PHS_001420"],
    "father": ["M-NX-D-005-0301"],
    "soldier": ["N0245-02-29-00", "7569_G1A2E7_KJK_003070", "N0001-06-29-00", "N0231-09-29-08"],
    "male": ["N0204-10-31-00", "D-NX-F-003-0201", "1658_G1A2E7_HMK_001674", "0042_G1A3E7S4C0_CHS_001786"],
    "female": ["N0169-05-30-00", "9027_G2A1E7_HJS_001223", "9013_G2A1E7_PYE_000237", "5817_G2A2E7_KSI_002989", "4503_G2A2E7_LMN_003875", "5817_G2A2E7_KSI_002783"],
    "female_child": ["A-NX-D-010-0051", "0033_G2A3E7S0C2_KMA_001625" ,"0033_G2A3E2S0C3_KMA_001680"],
}

script_1 = [
    {'role': 'heroine', 'style': '근엄하게', 'text': '긍지 높던 황금의 왕국, 로이몬드'},
    {'role': 'heroine', 'style': '걱정', 'text': '어느 날... 왕의 과욕으로 인해 찬란했던 영광은 사라지고 한순간에 무너져내렸는데...'},
    {'role': 'soldier', 'style': '큰소리로', 'text': '“잡아라!”'},
    {'role': 'heroine', 'style': '다급하게', 'text': '적들의 맹렬한 추격 속 죽기살기로 도망치고 있는 나, 제르이네'},
    {'role': 'heroine', 'style': '근엄하게', 'text': '나는 왕국의 여덟 번째 왕녀다'},
    {'role': 'heroine', 'style': '속이타듯', 'text': '언젠가 왕국을 다시 일으키려면, 후계자인 언니들이 꼭 살아남아야 하는데...'},
    {'role': 'heroine', 'style': '안심시키듯', 'text': '이를 위해선, 내가 미끼가 되어야겠지...'},
    {'role': 'heroine', 'style': '비장한', 'text': '하지만 나는… 살고 싶어! 이렇게 허무하게 죽을 순 없어!'},
    {'role': 'heroine', 'style': '다급하게', 'text': '그 순간...'},
    {'role': 'soldier', 'style': '다급하게', 'text': '“왕녀님! 조심…!”'},
    {'role': 'heroine', 'style': '충격받은', 'text': '나를 호위하던 기사는 눈앞에서 쓰러졌고'},
    {'role': 'heroine', 'style': '충격받은', 'text': '날아온 화살은 내가 탄 말에 박히고 말았다'},
    {'role': 'heroine', 'style': '절망한듯', 'text': '흙바닥에 내동댕이쳐지며 의식이 흐려졌는데...'},
    {'role': 'heroine', 'style': '어리둥절한듯', 'text': '얼마나 지났을까, 정신을 차려보니... 따스한 햇살과... 감옥?!'},
    {'role': 'heroine', 'style': '혼란스러운', 'text': '이상한데? 무슨 감옥이 밖에 있어?!'},
    {'role': 'heroine', 'style': '당황한듯', 'text': '몸을 일으키려다 균형을 잃고 발을 헛디뎠는데...'},
    {'role': 'heroine', 'style': '궁금한듯', 'text': '발밑에는 웬 막대기 하나가...?'},
    {'role': 'heroine', 'style': '멍한', 'text': '잠이 덜 깼나 싶어 눈을 비비려했더니'},
    {'role': 'heroine', 'style': '충격받은', 'text': '내 손 대신 퍼덕이는... 깃털 달린 날개?!'},
    {'role': 'heroine', 'style': '충격받은', 'text': '믿을 수 없어...! 내가… 내가 새라니?!'},
]
script_2 = [
    {'role': 'heroine', 'style': '절망한듯', 'text': '이거, 꿈이 아니잖아!, 내 삶을 돌려달라고!'},
    {'role': 'heroine', 'style': '혼란스러운', 'text': '현실을 부정하며 발버둥 치던 중, 한 병사가 다가와 나를 살피기 시작했다'},
    {'role': 'soldier', 'style': '궁금한듯', 'text': '“흐음... 얘가 왜 이러지? 아픈가? 보고를 올려야 하나?”'},
    {'role': 'heroine', 'style': '떠보듯이', 'text': '보고...? 일단 나는 이 사람의 새는 아닌가 보군!'},
    {'role': 'heroine', 'style': '비장한', 'text': '이대로 잡혀있을 순 없지!', 'prompt_idx': 2},
    {'role': 'heroine', 'style': '비장한', 'text': '나는 죽은 척 연기하며 기회를 노렸고,', 'prompt_idx': 2},
    {'role': 'heroine', 'style': '비장한', 'text': '이내 발톱 맛을 보여주며 탈출했다', 'prompt_idx': 2},
    {'role': 'heroine', 'style': '흥분한듯', 'text': '성공했어!, 내가 해냈다고!'},
    {'role': 'heroine', 'style': '구연체', 'text': '신나게 날아다니다 숲에 도착했는데...', 'prompt_idx': 1},
    {'role': 'heroine', 'style': '충격받은', 'text': '우리 왕국의 깃발과... 사람...?', 'prompt_idx': 1},
    #{'role': 'heroine', 'style': '충격받은', 'text': '우리 왕국의 깃발과... 사람...?!', 'prompt_idx': 1},
    {'role': 'heroine', 'style': '다급하게', 'text': '쓰러진 아군 전령병을 발견했고,'},
    {'role': 'heroine', 'style': '다급하게', 'text': '그의 품에는 왕국의 운명이 걸린 기밀문이 있었다'},
    #{'role': 'heroine', 'style': '다급하게', 'text': '게다가, 심장이 뛰잖아...?!, 이대로 두면 죽을 거야...!'},
    {'role': 'heroine', 'style': '다급하게', 'text': '게다가, 심장이 뛰잖아...?, 이대로 두면 죽을 거야...!'},
    {'role': 'heroine', 'style': '비장한', 'text': '나는 왕녀니까, 병사를 구하고 문서를 전달해야만 해!', 'prompt_idx': 2},
    {'role': 'heroine', 'style': '다급하게', 'text': '필사적으로 찾아 헤맨 끝에 마침내 아군진지를 발견했고,'},
    {'role': 'heroine', 'style': '다급하게', 'text': '나는 그곳을 향해 전속력으로 돌진했다'},
    @{'role': 'heroine', 'style': '당황한듯', 'text': '그런데 잠깐, 어... 어떻게 멈추는 거더라...?'},
    {'role': 'heroine', 'style': '다급하게', 'text': '속도 조절에 실패한 나는 결국 한 남자에게 부딪히기 직전,'},
    {'role': 'heroine', 'style': '걱정', 'text': '그의 손에 붙잡히고 말았는데...', 'prompt_idx': 1},
    {'role': 'hero', 'style': '냉정하게', 'text': '“짐승도 암살에 쓰나?”'},
]

script_3 = [
    {'role': 'heroine', 'style': '당황한듯', 'text': '아니, 왜 하필 제일 높은 사람한테 돌진한 거냐고!'},
    {'role': 'heroine', 'style': '두려운듯', 'text': '하필 날 붙잡은 건 왕국의 세 군대를 통솔하는 냉혹한 총사령관, 발하일...'},
    {'role': 'heroine', 'style': '억울한듯', 'text': '그는 내가 아군새라는 말에도 자신을 공격하려 했다며'},
    {'role': 'heroine', 'style': '의심스러운', 'text': '나를 적군의 스파이로 끝없이 의심했다'},
    {'role': 'heroine', 'style': '절망한듯', 'text': '결국 난 그의 막사 앞에 꼼짝없이 묶이게 된 상황!'},
    {'role': 'hero', 'style': '냉정하게', 'text': '"우연이라기엔 지나치게 이상하다"'},
    {'role': 'hero', 'style': '심각하게', 'text': '"어떻게 훈련도 받지 못한 새가 기밀문서를 들고 정확히 나에게 온 거지?"'},
    {'role': 'heroine', 'style': '억울한듯', 'text': '그는 심지어 기밀문이 조작됐을 가능성까지 파고들었는데...'},
    {'role': 'heroine', 'style': '답답한듯', 'text': '말만 할 수 있다면 다 설명할 텐데! 답답해 미치겠네!'},
    {'role': 'heroine', 'style': '비장한', 'text': '다급해진 나는, 최후의 수단을 쓰기로 결심했다'},
    {'role': 'heroine', 'style': '단호하게', 'text': '말이 안 통하면 몸으로 보여주는 수밖에!'},
    {'role': 'heroine', 'style': '발랄하게', 'text': '빙글빙글, 파닥파닥! 이 정도면 완벽한 설명 아니야?'},
    {'role': 'heroine', 'style': '간절하게', 'text': '제발, 내 진짜 뜻을 좀 이해해 보라고!'},
    {'role': 'heroine', 'style': '희망하는듯', 'text': '그때, 부관이 내 간절한 몸짓을 알아챈 듯 달려갔다'},
    {'role': 'heroine', 'style': '안도하는', 'text': '그래, 바로 그거야! 어서 보고해!'},
    {'role': 'heroine', 'style': '기대하는듯', 'text': '드디어 이 답답함이 풀리나 기대하던 그 순간,'},
    {'role': 'soldier', 'style': '단호하게', 'text': '"저 새는 미쳤습니다. 확실히!"'},
    {'role': 'heroine', 'style': '충격받은', 'text': '뭐라고오?!'},
]

script_4 = [
    {'role': 'heroine', 'style': '짜증낸듯', 'text': '명색이 왕녀인데, 미치고 팔짝 뛰겠네!'},
    {'role': 'heroine', 'style': '안도하는', 'text': '다행히 내 필사적인 몸짓을 본 발하일이 명령을 내렸다'},
    {'role': 'hero', 'style': '냉정하게', 'text': '“수색해“, “확실히... 정상이 아닌 것 같긴 하군”'},
    {'role': 'heroine', 'style': '억울한듯', 'text': '기껏 정보를 전해줬더니 미친 새 취급이라니!'},
    {'role': 'heroine', 'style': '참는듯', 'text': '참자, 여기서 찍히면 굶어 죽을지도 몰라!'},
    {'role': 'heroine', 'style': '혐오스러운', 'text': '그런데... 식사랍시고 주는 게 쥐와 개구리 한 사발?!'},
    {'role': 'heroine', 'style': '성가신', 'text': '치워! 난 엄연히 인간이라고!'},
    {'role': 'heroine', 'style': '조심스러운', 'text': '겨우 진정하고 발하일의 막사로 불려 갔는데...'},
    {'role': 'heroine', 'style': '환멸을 느끼는', 'text': '또! 또 그 끔찍한 세트잖아!'},
    {'role': 'heroine', 'style': '충격받은', 'text': '나도 모르게 그만, 질색하며 밥그릇을 뻥 차버렸다'},
    {'role': 'heroine', 'style': '간절하게', 'text': '나는 저걸 먹고 싶다고!'},
    {'role': 'heroine', 'style': '놀란듯', 'text': '혼날 줄 알았는데, 내 손짓을 본 발하일이 무심하게 빵 조각을 던져주었다'},
    {'role': 'heroine', 'style': '안도하는', 'text': '냉큼 받아먹고 책상을 뒤적이던 그 순간,'},
    {'role': 'hero', 'style': '심각하게', 'text': '“이걸 찾나?“'},
    {'role': 'heroine', 'style': '다급하게', 'text': '그의 손에 들린 기밀문!'},
    {'role': 'heroine', 'style': '초조한듯', 'text': '내용을 알아내기 위해 문서로 돌진했지만 그는 순식간에 숨겨버렸는데...'},
    {'role': 'heroine', 'style': '분노한듯', 'text': '고작 고기로 나를 회유하겠다고?'},
    {'role': 'heroine', 'style': '혼란스러운', 'text': '절대 안 먹을 거야!, ...아, 너무 맛있다!'},
    {'role': 'heroine', 'style': '놀란듯', 'text': '그런 나를 보더니 그가 하는 말이...,'},
    {'role': 'hero', 'style': '무심한듯', 'text': '“집착하지 마라”'},
]

script_5 = [
    {'role': 'heroine', 'style': '분노한듯', 'text': '아니, 집착이 아니라 가족과 내 생사를 확인해야 한다고!'},
    {'role': 'heroine', 'style': '억울한듯', 'text': '나는 기밀문서를 엿보려다 식탐부리는 새로 오해받고,'},
    {'role': 'heroine', 'style': '짜증낸듯', 'text': '그는 새의 덕목을 운운하며 나를 다그쳤다'},
    {'role': 'heroine', 'style': '화가난듯', 'text': '억울함에 성질대로 소리 지르고 분노의 부리질을 해봤지만...'},
    {'role': 'heroine', 'style': '낙담한', 'text': '내게 돌아온 건 고된 전령새 훈련 뿐!'},
    {'role': 'heroine', 'style': '체념한듯', 'text': '나는 매일 나무토막이나 물어 나르는 신세가 되었는데...'},
    {'role': 'heroine', 'style': '궁금한듯', 'text': '그러던 중 병사들의 대화에서 한 줄기 희망을 발견했다'},
    {'role': 'heroine', 'style': '기쁜듯', 'text': '우리 왕족을 찾고 있다는 소식!'},
    {'role': 'heroine', 'style': '희망적인', 'text': '그래, 아직 끝난 게 아니었어! 분명 누군가 살아있는거야!'},
    {'role': 'heroine', 'style': '초조한듯', 'text': '그렇게 버티던 어느 날, 닷새 만에 돌아온 발하일이 충격적인 말을 꺼냈다'},
    {'role': 'hero', 'style': '냉정하게', 'text': '"전령새 대부분이 돌아오지 않았다, 이제 남은 전령새는 이 새 하나다"'},
    {'role': 'hero', 'style': '단호하게', 'text': '"...이 녀석을 보내려고 한다"'},
    {'role': 'heroine', 'style': '두려운듯', 'text': '갑자기 훈련도 덜 끝난 나를 위험한 임무에 보내겠다는데...'},
    {'role': 'soldier', 'style': '궁금한듯', 'text': '"무슨 전갈을 보내려고 하십니까?"'},
    {'role': 'hero', 'style': '의미심장한', 'text': '그런 일이 있다'},
    {'role': 'heroine', 'style': '혼란스러운', 'text': '그의 최측근조차 모르는 극비 임무라니...'},
    {'role': 'heroine', 'style': '비장한', 'text': '어쩌면, 이게 나라를 구할 수 있는 기회일지도 몰라!'},
    {'role': 'heroine', 'style': '단호하게', 'text': '"좋아, 하겠어요, 그 임무!"'},
]

def prefix_tag(tag, text):
    return f"<|tag_start|>{tag}<|tag_end|>{text}"

llm_path="/home/longtou.2024/mount/longtou/h100/exp/cosyvoice/20250908/torch_ddp/epoch_1_step_70000.pt"
flow_path=None
cosyvoice = CosyVoice2('/home/longtou.2024/mount/longtou/saved/cosyvoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False, llm_path=llm_path, flow_path=flow_path)

outdir = "outdir"
Path(outdir).mkdir(exist_ok=True)

prompt_spk_speech_16k = dict()
for audio_path in Path("test/shorts/shorts_wavs_lt").glob("**/*.wav"):
    uttid = audio_path.stem
    prompt_spk_speech_16k[uttid] = load_wav(audio_path, 16000)
prompt_spk_speech_24k = dict()
for audio_path in Path("test/shorts/shorts_wavs_lt").glob("**/*.wav"):
    uttid = audio_path.stem
    prompt_spk_speech_24k[uttid] = load_wav(audio_path, 24000)

prompt_spk_sent = dict()
for json_path in Path("test/shorts/shorts_wavs_lt").glob("**/*.json"):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    uttid = json_path.stem
    if "transcript" in json_data:
        transcript = json_data["transcript"].strip()
    elif "voice_piece" in json_data:
        transcript = json_data["voice_piece"]["tr"].strip()
    elif "text_info" in json_data:
        transcript = json_data["text_info"]["OrgLabelText"].strip()
    elif "전사정보" in json_data:
        transcript = json_data["전사정보"]["OrgLabelText"].strip()
    else:
        raise Exception
    prompt_spk_sent[uttid] = transcript

female_num = 5
male_num = 0
soldier_num = 3
role = 'heroine'
tts_speech = []
tts_uttid = set()
for line in script_2:
    #if role == 'heroine':
    #    uttid = spk_roles['female'][female_num]
    #elif role == 'hero':
    #    uttid = spk_roles['male'][male_num]
    #elif role == 'soldier':
    #    uttid = spk_roles['soldier'][soldier_num]
    #else:
    #    raise Exception(role)
    #prompt_sent = prompt_spk_sent[uttid]
    #prompt_speech_16k = prompt_spk_speech_16k[uttid]
    #prompt_speech_24k = prompt_spk_speech_24k[uttid]

    tag = line['style']
    this_text = line['text']

    prompt_idx = line.get("prompt_idx", 0)
    prompt_path = f"test/tag_zs/outdir/5817_G2A2E7_KSI_002783/{tag}/{prompt_idx}"
    with open(prompt_path + ".txt", 'r') as f:
        lines = f.readlines()
    prompt_sent = lines[0].strip()
    prompt_speech_16k = load_wav(prompt_path + ".wav", 16000)
    prompt_speech_24k = load_wav(prompt_path + ".wav", 24000)

    #gen = cosyvoice.inference_zero_shot(prefix_tag(tag, this_text), prompt_sent, prompt_speech_16k, stream=False, text_frontend=False, prompt_speech_24k=prompt_speech_24k)
    gen = cosyvoice.inference_zero_shot(f" {this_text}", prompt_sent, prompt_speech_16k, stream=False, text_frontend=False, prompt_speech_24k=prompt_speech_24k)
    ret = next(gen)
    tts_speech.append(ret['tts_speech'])
tts_speech = torch.cat(tts_speech, dim=1)
torchaudio.save(f"{outdir}/debug.wav", tts_speech, cosyvoice.sample_rate)

