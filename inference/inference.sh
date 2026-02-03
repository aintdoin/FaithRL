set -x
export CUDA_VISIBLE_DEVICES=5
MODEL_NAME=qwen-rlcr
export MODEL_TEMPLATE=llama
DATASET=2wikimultihop
MODEL_PATH=checkpoints/RLCR
#Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct
test_files="['data/$DATASET/test_false.parquet']"
export SYSTEM_PROMPT_TYPE=rlcr #idk_aware, idk_not_aware, rlcr, directly
OUTPUT_FILE=inference/inference_results/$MODEL_NAME/$DATASET.jsonl
FILTER_TYPE=all  # all, answerable, unanswerable
NUM_SAMPLES=-1  
MAX_MODEL_LEN=16384  
TENSOR_PARALLEL_SIZE=1
TEMPERATURE=0
TOP_P=1.0  # Ignored in greedy mode
TOP_K=-1   # Ignored in greedy mode  
REPETITION_PENALTY=1.0  # No repetition penalty in validation
MAX_TOKENS=4096  
# LLM Judge configuration (must match server configuration)
export USE_LLM_JUDGE=true
export LLM_JUDGE_API_BASE=http://localhost:8000
export LLM_JUDGE_MODEL_NAME=
export LLM_JUDGE_API_KEY=
export LLM_JUDGE_MAX_WORKERS=8
export LLM_JUDGE_TIMEOUT=60
python inference/inference.py \
    --test-files "$test_files" \
    --output-dir "$OUTPUT_FILE" \
    --model-path $MODEL_PATH \
    --model-name $MODEL_NAME \
    --filter-type $FILTER_TYPE \
    --num-samples $NUM_SAMPLES \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --repetition-penalty $REPETITION_PENALTY \
    --max-tokens $MAX_TOKENS \
    ${CHECKPOINT_PATH:+--checkpoint-path "$CHECKPOINT_PATH"} \
    $@