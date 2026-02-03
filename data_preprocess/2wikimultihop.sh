# type: train, test
# template_type: deepseek-r1-distill-qwen, deepseek-r1-distill-llama, base, qwen-instruct, llama-instruct
export USE_LLM_JUDGE=true  # Enable LLM judge
export CUDA_VISIBLE_DEVICES=0,1,2,3
# LLM Judge API Configuration (renamed from ANSWER_EXTRACT_* for clarity)
export LLM_JUDGE_API_BASE=http://localhost:8000  
export LLM_JUDGE_MODEL_NAME=
export LLM_JUDGE_API_KEY=  # Empty or your API key
LOG_FILE="./logs/2wikimultihop_process.log"
mkdir -p "$(dirname "$LOG_FILE")"
echo "Starting 2wikimultihop preprocessing with LLM judge..."
echo "Logs will be saved to: $LOG_FILE"
exec nohup python data_preprocess/2wikimultihop.py \
    --type train \
    --template_type qwen \
    --size 15000 \
    --data-path ./2wikimultihop/data/train.jsonl \
    --model-path \
    --tensor-parallel-size 4 \
    --n-candidates 32 \
    --temperature 1.0 --top-p 0.95 --top-k 100 --max-tokens 2048 > "$LOG_FILE" 2>&1 &
echo "Process started in background. PID: $!"