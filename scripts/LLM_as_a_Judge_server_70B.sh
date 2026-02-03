set -euo pipefail
LOG_FILE="${LOG_FILE:-./logs/LLM_as_a_Judge_server.log}"
PID_FILE="${PID_FILE:-./logs/LLM_as_a_Judge_server.pid}"
MODEL_PATH=
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096} 
GPU_UTIL=${GPU_UTIL:-0.85}
export CUDA_VISIBLE_DEVICES=6,7
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}  
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}  
if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is required" >&2
  exit 1
fi
echo "Starting vLLM OpenAI server on ${HOST}:${PORT} with model ${MODEL_PATH}..."
echo "Using ${TENSOR_PARALLEL_SIZE} GPUs for tensor parallelism (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Logs will be saved to: ${LOG_FILE}"
echo "PID will be saved to: ${PID_FILE}"
echo "Batch settings: max_num_seqs=${MAX_NUM_SEQS}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
mkdir -p "$(dirname "${LOG_FILE}")"
mkdir -p "$(dirname "${PID_FILE}")"
nohup python3 -m vllm.entrypoints.openai.api_server \
  --host ${HOST} \
  --port ${PORT} \
  --model ${MODEL_PATH} \
  --max-model-len ${MAX_MODEL_LEN} \
  --gpu-memory-utilization ${GPU_UTIL} \
  --dtype bfloat16 \
  --trust-remote-code \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --enable-prefix-caching \
  --max-num-seqs ${MAX_NUM_SEQS} \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} > "${LOG_FILE}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"
echo "vLLM server started. PID=${SERVER_PID}"