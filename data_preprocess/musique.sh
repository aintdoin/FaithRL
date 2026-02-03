# type: train, test
# template_type: deepseek-r1-distill-qwen, deepseek-r1-distill-llama, base, qwen-instruct, llama-instruct
export LLM_JUDGE_API_BASE=http://localhost:8000
export LLM_JUDGE_MODEL_NAME=
export LLM_JUDGE_API_KEY=
#python -m data_preprocess.musique --type test --size 2000
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -m data_preprocess.musique --type train --size 39876 --log-every 100 > musique_data_train.log 2>&1 &