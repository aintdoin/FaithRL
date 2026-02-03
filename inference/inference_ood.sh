set -x
export CUDA_VISIBLE_DEVICES=5
MODEL_PATH=checkpoints/qwen_7b/GRPO/global_step_310/actor/huggingface
MODEL_NAME=qwen_7b
export MODEL_TEMPLATE=qwen
TEST_FILES="['data/GPQA_diamond/test.jsonl', 'data/GSM8k/test.jsonl', 'data/MATH500/test.jsonl']"
OUTPUT_DIR="inference/results_ood/$MODEL_NAME/GRPO"
NUM_SAMPLES=-1
MAX_MODEL_LEN=4096
TENSOR_PARALLEL_SIZE=1
TEMPERATURE=0
MAX_TOKENS=4096
export LLM_JUDGE_API_BASE=http://localhost:8000
export LLM_JUDGE_MODEL_NAME=
export LLM_JUDGE_API_KEY=
export LLM_JUDGE_MAX_WORKERS=16
export LLM_JUDGE_TIMEOUT=60
python inference/inference_ood.py \
    --test-files "$TEST_FILES" \
    --output-dir "$OUTPUT_DIR" \
    --model-path "$MODEL_PATH" \
    --model-name "$MODEL_NAME" \
    --num-samples $NUM_SAMPLES \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --temperature $TEMPERATURE \
    --max-tokens $MAX_TOKENS \
    $@