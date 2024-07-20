## 1  environment setup
```
cd release
conda create --name <env> --file requirement.txt
```

## 2  data preprocess
```
bash script/prepare_train_data.sh 
```

## 3 warmup training
TODO:automatic train 4 epochs while mannual stop at 30 steps needed.
```
bash script/train/warmup_lora_train.sh ./data meta-llama/Llama-2-7b-hf 0.05 3 llama2-7b-p0.05-lora-seed3
```

## 4 gradient data store
```
CUDA_VISIBLE_DEVICES=0 bash script/prepare_grad_data.sh
```

## 5 select data 
```
python3 -m utils.mean_grad --gradient_path grads_full_data/llama2-7b-p0.05-lora-seed3/{}-ckpt{}-adam/dim8192/all_orig.pt --train_file_names hard_coded lima code_alpaca baize cot dolly flan_v2 gpt4_alpaca oasst1 open_orca science self_instruct sharegpt stanford_alpaca super_ni unnatural_instructions wizardlm --ckpts 10 20 30 --checkpoint_weights 2.1569e-06 6.0784e-06 1e-05 --output_path selected_full_data_102030

python3 -m utils.sk_LESS

python3 -m utils.omp out/cluster_LESS_102030_100.pkl out/center_LESS_102030_100.pkl out/top_omp_LESS102030_100.pkl

python3 -m utils.get_segment_data
```

## 6 train with seleted data
```
bash ./script/train/lora_train.sh out/top_omp_LESS102030_100_data.jsonl meta-llama/Llama-2-7b-hf llama2-7b-0.05-omp-LESS102030-100-lora 
```

## 7 eval
Please follow the instructions in the evaluation folder to evaluate the performance of the model trained on the selected data by using open-instruct.
