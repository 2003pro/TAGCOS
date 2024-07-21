cd ../open-instruct
python -m eval.bbh.run_eval --data_dir data/eval/ --save_dir test/test_res_bbh_LESS102030_100_omp_0.05 --model out/llama2-7b-0.05-omp-LESS102030-100-lora  --tokenizer out/llama2-7b-0.05-omp-LESS102030-100-lora  --max_num_examples_per_task 40 --eval_batch_size 20
