cd ../open-instruct
python -m eval.mmlu.run_eval --data_dir data/eval/mmlu --ntrain 5  --save_dir test/test_res_mmlu_LESS102030_100_omp_0.05  --model_name_or_path out/llama2-7b-0.05-omp-LESS102030-100-lora  --tokenizer_name_or_path out/llama2-7b-0.05-omp-LESS102030-100-lora  --eval_batch_size 16
