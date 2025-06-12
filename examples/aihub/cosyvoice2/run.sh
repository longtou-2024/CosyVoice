#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

export OMP_NUM_THREADS=1
stage=1
stop_stage=5
pretrained_model_dir=/home/longtou.2024/mount/longtou/saved/cosyvoice/pretrained_models/CosyVoice2-0.5B
num_workers=1
prefetch=100
cache_size=0
from_mount=true
conf=conf/cosyvoice2_lt.yaml # DO NOT USE 'CONFIG', its var name is used in 'parse_options.sh'
train_data="gs://literature skt_emotion_large skt_emotion_small mediazen_emotion mediazen commbooks aihub_news mediazen_adult mediazen_teen saltlux_jeju saltlux_chungcheong saltlux_gyeongsang saltlux_jeolla saltlux_gangwon emilia_ko emilia_yodas_ko emilia_en emilia_zh"
#train_data="gs://literature"
cv_data="gs://azure"
train_engine=torch_ddp
model_dir=`pwd`/exp/cosyvoice2/llm
tensorboard_dir=`pwd`/tensorboard/cosyvoice2/llm
deepspeed_config=./conf/ds_stage2.json
checkpoint= # llm checkpoint for resume training

. parse_options.sh

model_dir=$model_dir/$train_engine
tensorboard_dir=$tensorboard_dir/$train_engine
decode_checkpoint=$model_dir/llm_avg.pt
src_path=$model_dir/tobe_averaged

_opts=
if [ -n "$checkpoint" ]; then
  _opts+="--checkpoint ${checkpoint}"
fi
if [ "${from_mount}" = true ]; then
  _opts+=" --from_mount"
fi

# train llm
#export CUDA_VISIBLE_DEVICES="0,1"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  torchrun --nnodes=1 --nproc_per_node=$num_gpus \
      --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
    cosyvoice/bin/train.py \
    --train_engine $train_engine \
    --config ${conf} \
    --train_data "${train_data}" \
    --cv_data "${cv_data}" \
    --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
    --model llm \
    --model_dir ${model_dir} \
    --tensorboard_dir ${tensorboard_dir} \
    --ddp.dist_backend $dist_backend \
    --num_workers ${num_workers} \
    --prefetch ${prefetch} \
    --pin_memory \
    --deepspeed_config ${deepspeed_config} \
    --deepspeed.save_states model+optimizer \
    --cache_size ${cache_size} \
    ${_opts}
fi
  #--use_amp \ # infinity grad norm error?
#--checkpoint $pretrained_model_dir/$model.pt \

# average model
average_num=1
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "do model average and final checkpoint is $decode_checkpoint"
  python cosyvoice/bin/average_model.py \
    --dst_model $decode_checkpoint \
    --src_path $src_path  \
    --num ${average_num} \
    --val_best
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi
