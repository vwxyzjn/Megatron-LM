#!/bin/bash

# Runs DeepSeek V3 model

# DeepSeek V3 specific environment variables
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export PYTHONWARNINGS=ignore
export NCCL_DEBUG=VERSION
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=${1:-"checkpoints/deepseek_v3"}
TOKENIZER_MODEL=${3:-"MOCK"}
DATA_ARG=${4:-"MOCK"}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-2048}
mkdir -p "$(dirname "$CHECKPOINT_PATH")"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --sequence-parallel
    --use-flash-attn
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 4096
    --num-layers 61
    --hidden-size 7168
    --ffn-hidden-size 18432
    --num-attention-heads 128
    --kv-channels 128
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --clip-grad 1.0
    --weight-decay 0.1
    --qk-layernorm
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --position-embedding-type rope
    --rotary-base 10000
    --rotary-scaling-factor 40
    --mscale 1.0
    --mscale-all-dim 1.0
    --swiglu
    --untie-embeddings-and-output-weights
    --multi-latent-attention
    --no-masked-softmax-fusion
    --no-position-embedding
    --make-vocab-size-divisible-by 3232
    --transformer-impl transformer_engine
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --manual-gc
    --manual-gc-interval 10
    --no-check-for-nan-in-loss-and-grad
)

# MLA (Multi-Latent Attention) arguments
MLA_ARGS=(
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
)

MOE_ARGS=(
    --num-experts 256
    --moe-layer-freq "([0]*3+[1]*58)"
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk 8
    --moe-token-dispatcher-type flex
    --moe-enable-deepep
    --moe-router-pre-softmax
    --moe-grouped-gemm
    --moe-aux-loss-coeff 1e-4
    --moe-router-group-topk 4
    --moe-router-num-groups 8
    --moe-router-topk-scaling-factor 2.5
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-dtype fp32
    --moe-permute-fusion
)

# Data arguments (conditional for mock vs real data)
# DATA_ARGS_LIST=()
# if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]] || [[ -z "$TOKENIZER_ARG" ]]; then
#     DATA_ARGS_LIST+=(
#         "--mock-data"
#         "--train-samples $TRAIN_SAMPLES"
#         "--lr-decay-samples 584765624"
#         "--lr-warmup-samples 1536000"
#         "--tokenizer-type NullTokenizer"
#         "--vocab-size 128256" 
#         "--tiktoken-pattern v2" 
#         "--split '99,1,0'"
#         "--no-create-attention-mask-in-dataloader"
#         "--no-mmap-bin-files"
#         "--num-workers 6"
#     )
# else
#     # Settings for real data with DeepSeek V3 tokenizer
#     DATA_ARGS_LIST+=(
#         "--data-path $DATA_ARG"
#         "--tokenizer-type HuggingFaceTokenizer" 
#         "--tokenizer-model deepseek-ai/DeepSeek-V3"
#         "--split '99,1,0'"
#         "--no-create-attention-mask-in-dataloader"
#         "--no-mmap-bin-files"
#         "--num-workers 6"
#         "--make-vocab-size-divisible-by 3232"
#     )
# fi

    # Settings for real data with DeepSeek V3 tokenizer
DATA_ARGS_LIST+=(
    "--mock-data"
    "--train-samples $TRAIN_SAMPLES"
    "--tokenizer-type HuggingFaceTokenizer" 
    "--tokenizer-model deepseek-ai/DeepSeek-V3"
    "--split '99,1,0'"
    "--no-create-attention-mask-in-dataloader"
    "--no-mmap-bin-files"
    "--num-workers 6"
    "--make-vocab-size-divisible-by 3232"
)
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8192
    --lr-warmup-init 3.9e-7
    --lr 3.9e-6
    --min-lr 3.9e-7
    --lr-decay-style cosine
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --exit-duration-in-mins 220
    --no-save-optim
    --distributed-timeout-minutes 60
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 8
    --expert-model-parallel-size 64
    --context-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-distributed-optimizer
    --enable-experimental
)

LOGGING_ARGS=(
    --log-interval 1
    --save-interval 500
    --eval-interval 200
    --eval-iters 32
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    --log-throughput
    --log-timers-to-tensorboard false
    --log-memory-to-tensorboard true
    --log-num-zeros-in-grad false
    --log-params-norm false
    --log-validation-ppl-to-tensorboard true
    --logging-level 40
    --no-load-optim
    --no-load-rng
    --finetune false
    --auto-detect-ckpt-format
    --dist-ckpt-strictness log_all
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"DeepSeek-V3"}
        --wandb-exp-name ${WANDB_NAME:-"DeepSeek-V3"}
    )
fi


torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MLA_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
