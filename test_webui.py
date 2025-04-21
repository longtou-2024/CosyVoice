# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
import librosa
import random

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

import gradio as gr

from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['sft', '3s_clonning', 'cross_lingual', 'instruct_infer']
instruct_dict = {'sft': '1. Choose pretrained tone\n2. Clike to generate audio button',
                 '3s_clonning': '1. Choose a prompt audio file or enter a prompt audio file, with a maximum duration of 30 seconds. If provided simultaneously, prioritize the prompt audio file.\n2. Input prompt text\n3. Click the generate audio button',
                 'cross_lingual': '1. Choose a prompt audio file or enter a prompt audio file, with a maximum duration of 30 seconds. If provided simultaneously, prioritize the prompt audio file.\n2. Click the generate audio button',
                 'instruct_infer': '1. Choose pretrained tone\n2. Input instruction text\n3. Click the generate audio button'}
stream_mode_list = [('no', False), ('yes', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['instruct_infer']:
        if cosyvoice.instruct is False:
            gr.Warning('You are using natural language control mode, but the {} model does not support this mode. Please use the iic/CosyVoice-300M-Instruct model'.format(args.model_dir))
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text == '':
            gr.Warning('You are using natural language control mode. Please enter the instruction text.')
            yield (cosyvoice.sample_rate, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('You are using natural language control mode, so prompt audio/prompt text will be ignored.')
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['cross_lingual']:
        if cosyvoice.instruct is True:
            gr.Warning('You are using cross-language replication mode. The {} model does not support this mode, please use the iic/CosyVoice-300M model.'.format(args.model_dir))
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('You are using cross-language replication mode, and the instruction text will be ignored.')
        if prompt_wav is None:
            gr.Warning('You are using cross-language replay mode. Please provide prompt audio.')
            yield (cosyvoice.sample_rate, default_data)
        gr.Info('You are using cross-language replication mode. Please ensure that the synthesized text and prompt text are in different languages.')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s_clonning', 'cross_lingual']:
        if prompt_wav is None:
            gr.Warning('Prompt audio is empty. Have you forgotten to input prompt audio?')
            yield (cosyvoice.sample_rate, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('prompt audio sampling rate {} below {}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (cosyvoice.sample_rate, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['sft']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('You are using the pre-trained tone mode, and prompt text/prompt audio/instruct text will be ignored!')
        if sft_dropdown == '':
            gr.Warning('No pre-trained sounds available!')
            yield (cosyvoice.sample_rate, default_data)
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s_clonning']:
        if prompt_text == '':
            gr.Warning('The prompt text is empty. Have you forgotten to input the prompt text?')
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('You are using 3s_clonning, and the sft/instruct text will be ignored!')

    if mode_checkbox_group == 'sft':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '3s_clonning':
        logging.info('get zero_shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        ###
        def tts_text_gen():
            yield tts_text
        ###
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
        #for i in cosyvoice.inference_zero_shot_rt(tts_text_gen(), prompt_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == 'cross_lingual':
        logging.info('get cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    预训练模型 [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### Enter the text you want to synthesize, select the inference mode, and follow the prompts.")

        tts_text = gr.Textbox(label="Input synthetic text", lines=1, value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='Choose Reasoning Mode', value=inference_mode_list[0])
            instruction_text = gr.Text(label="Operation steps", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='Choose Pre-trained Tone', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='Is it streaming inference?', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="Speed Regulation (Supports Non-Flow Reasoning Only)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="Random inference seed")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Choose a prompt audio file with a sampling rate of at least 16 kHz.')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Record prompt audio files')
        prompt_text = gr.Textbox(label="Input prompt text", lines=1, placeholder="Please enter prompt text that matches the prompt audio content; automatic recognition is currently not supported...", value='')
        instruct_text = gr.Textbox(label="Input instruction text", lines=1, placeholder="Input instruction text", value='')

        generate_button = gr.Button("Generate audio")

        audio_output = gr.Audio(label="Generate audio", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=46259)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice2(args.model_dir)
    #cosyvoice = CosyVoice2(args.model_dir, load_trt=True, use_flow_cache=True)

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
