# type: train, test
# template_type: deepseek-r1-distill-qwen, deepseek-r1-distill-llama, base, qwen-instruct, llama-instruct
python data_preprocess/hotpot_with_filter.py \
    --type train \
    --size 5000 \
    --template_type qwen \
    --model-path \
    --n-candidates 32 \
    --tensor-parallel-size 4
python data_preprocess/hotpot_with_filter.py \
    --type test \
    --size 2000 \
    --template_type qwen \
    --model-path  \
    --n-candidates 32 \
    --tensor-parallel-size 4