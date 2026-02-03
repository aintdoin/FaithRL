set -x
MODEL_PATH=
export TASK=musique #
export MODEL_NAME=qwen_7b  #qwen_7b 
export STRATEGY=grpo_evar_math_weighted #grpo_evar_math_weighted, knowrl, fspo, grpo
export SYSTEM_PROMPT_TYPE=idk_aware #idk_aware, idk_not_aware, rlcr, directly, r-tuning
export USE_REWARD=THS #GRPO, TRUTHRL, THS, RLCR, KNOWRL
export ALPHA=0.75 #fspo should be -1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_TEMPLATE=qwen
train_files="['data/musique/train.parquet']"
#train_files="['data/musique/train_true.parquet', 'data/musique/train_false.parquet']"
test_files="['data/2wikimultihop/test_true.parquet', 'data/2wikimultihop/test_false.parquet', 'data/hotpot/test_true.parquet', 'data/hotpot/test_false.parquet', 'data/musique/test_true.parquet', 'data/musique/test_false.parquet']"
REWARD_MODEL_PATH=./models/HHEM
export NLTK_DATA=./nltk_data
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export RAY_memory_usage_threshold=0.98
export USE_LLM_JUDGE=true
export LLM_JUDGE_API_BASE=http://localhost:8000
export LLM_JUDGE_MODEL_NAME=
export LLM_JUDGE_API_KEY=  # Empty or your API key
export LLM_JUDGE_MAX_WORKERS=8  # Number of concurrent judge requests
export LLM_JUDGE_TIMEOUT=60  # Timeout per request (seconds)
export GRPO_VARIANCE_THRESHOLD=0
export BETA_WARMUP_STEPS=100
# Reduce judge FLOPs for evar_math_weighted by using rule-based stepwise scoring
if [ "$STRATEGY" = "grpo_evar_math_weighted" ]; then
    export EVAR_REASONING_JUDGE_MODE=rule
fi
#export THS_X0=0.71226
#export THS_Y0=0.1365976
EXTRA_ARGS=""
if [ "$MODEL_TEMPLATE" = "llama" ]; then
    EXTRA_ARGS="+actor_rollout_ref.rollout.stop_token_ids=[128001,128009]"
fi
nohup python3 -m verl.trainer.main_ppo \
    $EXTRA_ARGS \
    actor_rollout_ref.actor.grad_clip=0.5 \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    +actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    data.shuffle=True \
    reward_model.enable=True \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.micro_batch_size_per_gpu=32 \
    reward_model.model.trust_remote_code=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$TASK \
    +trainer.use_ths_for_checkpoint=false \
    trainer.experiment_name=${STRATEGY}_${USE_REWARD}_${ALPHA} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir="checkpoints/${MODEL_NAME}/\${trainer.experiment_name}" \
    trainer.resume_mode='checkpoints/qwen_7b/grpo_evar_math_weighted_qwen_7b_0.75/global_step_230' \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=10 \
    trainer.total_epochs=1 $@ > ${STRATEGY}_${USE_REWARD}_${MODEL_NAME}_${ALPHA}.log 2>&1 &
    #trainer.resume_mode='checkpoints/qwen_7b/grpo_evar_math_weighted_qwen_7b_0.75/global_step_230' \
# MiniCheck API Configuration (replaces HHEM)
#export MINICHECK_API_BASE=http://localhost:8001
#export MINICHECK_MAX_WORKERS=8  # Number of concurrent API requests
#export MINICHECK_TIMEOUT=30  # Timeout per request (seconds)
# Filter out unanswerable samples during training when set to true
#export disable_unanswerable=false
# Legacy (DISABLED - no longer used)
#export USE_LLM_ANSWER_EXTRACTION=false  # Disabled - we don't do LLM extraction anymore
#export ENABLE_BATCH_ANSWER_PROCESSING=false  # Disabled - not needed for judge-only mode
#export IDK_PENALTY_ANSWERABLE=-0.3