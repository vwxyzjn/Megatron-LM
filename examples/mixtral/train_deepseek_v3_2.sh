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

# Parallelism and training hyperparameters from env (with sensible defaults)
TP=${TP:-2}
PP=${PP:-8}
EP=${EP:-64}
CP=${CP:-1}
VPP=${VPP:-1}
PP_FIRST=${PP_FIRST:-}
PP_LAST=${PP_LAST:-}
MBS=${MBS:-1}
GBS=${GBS:-2048}
SEQ_LEN=${SEQ_LEN:-4096}
COMMENT=${COMMENT:-""}

# IO paths
OUTPUT_PATH=${OUTPUT_PATH:-"checkpoints/deepseek_v3"}
LOAD_PATH=${LOAD_PATH:-"${OUTPUT_PATH}"}
DATA_PATH=${DATA_PATH:-"/path/to/your/data"}
WANDB_PROJECT=${WANDB_PROJECT:-"DeepSeek-V3"}

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
TRAIN_SAMPLES=${TRAIN_SAMPLES:-16384}
mkdir -p "$(dirname "$OUTPUT_PATH")"

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
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${SEQ_LEN}
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

# --moe-enable-deepep is not compatible with our EFA setup
MOE_ARGS=(
    --num-experts 256
    --moe-layer-freq "([0]*3+[1]*58)"
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk 8
    --moe-token-dispatcher-type flex
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

DATA_ARGS_LIST=(
    "--mock-data"
    "--train-samples $TRAIN_SAMPLES"
    "--tokenizer-type HuggingFaceTokenizer"
    "--tokenizer-model deepseek-ai/DeepSeek-V3"
    "--split 99,1,0"
    "--no-create-attention-mask-in-dataloader"
    "--no-mmap-bin-files"
    "--num-workers 6"
    "--make-vocab-size-divisible-by 3232"
)
TRAINING_ARGS=(
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --lr-decay-samples ${TRAIN_SAMPLES}
    --lr-warmup-samples 128
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
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
    --context-parallel-size ${CP}
    --expert-tensor-parallel-size 1
    --use-distributed-optimizer
    --enable-experimental
)

# Virtual pipeline parallelism
if [[ ${VPP} -gt 1 ]]; then
    MODEL_PARALLEL_ARGS+=(
        --num-virtual-stages-per-pipeline-rank ${VPP}
    )
fi

# Uneven pipeline parallelism layout if provided
if [[ -n "${PP_FIRST}" && -n "${PP_LAST}" ]]; then
    MODEL_PARALLEL_ARGS+=(
        --decoder-first-pipeline-num-layers ${PP_FIRST}
        --decoder-last-pipeline-num-layers ${PP_LAST}
    )
fi

LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 200 \
    --eval-iters 32 \
    --save ${OUTPUT_PATH} \
    --load ${LOAD_PATH} \
    --tensorboard-dir "${OUTPUT_PATH}/tensorboard" \
    --log-throughput \
    --no-load-optim \
    --logging-level 40 \
    --no-load-rng \
    --auto-detect-ckpt-format \
    --dist-ckpt-strictness log_all
)

if [ -n "${WANDB_API_KEY}" ]; then
    EXP_NAME=${WANDB_NAME:-"DeepSeek-V3-TP${TP}PP${PP}EP${EP}CP${CP}VPP${VPP}-MBS${MBS}GBS${GBS}-${COMMENT}"}
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT}
        --wandb-exp-name ${EXP_NAME}
    )
fi


torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MLA_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    "$@"
