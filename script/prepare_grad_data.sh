#!/bin/bash

# 定义数据集名称和文件路径
declare -A datasets
datasets["baize"]="data/train/processed/baize/baize_data.jsonl"
datasets["code_alpaca"]="data/train/processed/code_alpaca/code_alpaca_data.jsonl"
datasets["cot"]="data/train/processed/cot/cot_data.jsonl"
datasets["dolly"]="data/train/processed/dolly/dolly_data.jsonl"
datasets["flan_v2"]="data/train/processed/flan_v2/flan_v2_data.jsonl"
datasets["gpt4_alpaca"]="data/train/processed/gpt4_alpaca/gpt4_alpaca_data.jsonl"
datasets["hard_coded"]="data/train/processed/hard_coded/hard_coded_data.jsonl"
datasets["lima"]="data/train/processed/lima/lima_data.jsonl"
datasets["oasst1"]="data/train/processed/oasst1/oasst1_data.jsonl"
datasets["open_orca"]="data/train/processed/open_orca/open_orca_data.jsonl"
datasets["science"]="data/train/processed/science/science_data.jsonl"
datasets["self_instruct"]="data/train/processed/self_instruct/self_instruct_data.jsonl"
datasets["sharegpt"]="data/train/processed/sharegpt/sharegpt_data.jsonl"
datasets["stanford_alpaca"]="data/train/processed/stanford_alpaca/stanford_alpaca_data.jsonl"
datasets["super_ni"]="data/train/processed/super_ni/super_ni_data.jsonl"
datasets["tulu_v1"]="data/train/processed/tulu_v1/tulu_v1_data.jsonl"
datasets["tulu_v2"]="data/train/processed/tulu_v2/tulu_v2_data.jsonl"
datasets["unnatural_instructions"]="data/train/processed/unnatural_instructions/unnatural_instructions_data.jsonl"
datasets["wizardlm"]="data/train/processed/wizardlm/wizardlm_data.jsonl"

GRADIENT_TYPE="adam"
OUTPUT_BASE_PATH="../grads_full_data/llama2-7b-p0.05-lora-seed3"
DIMS="8192"

# 循环处理每个数据集
for CKPT in 10 20 30
do
  MODEL_BASE_PATH="out/llama2-7b-p0.05-lora-seed3-full_data/checkpoint-${CKPT}"

  for TRAINING_DATA_NAME in "${!datasets[@]}"
  do
    TRAINING_DATA_FILE=${datasets[$TRAINING_DATA_NAME]}
    MODEL_PATH="${MODEL_BASE_PATH}"
    OUTPUT_PATH="${OUTPUT_BASE_PATH}/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}"

    echo "Processing $TRAINING_DATA_NAME"
    ./script/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"
  done
done
