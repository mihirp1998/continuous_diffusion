#!/bin/bash

# Usage: ./run_multinode.sh 2,3,4,5 <command>
# Example: ./run_multinode.sh 1,2 main_jit.py exps=ocr_noise model=JiT-E-nb eval_freq=5 do_gen_perplexity=True online_eval=True

if [ $# -lt 2 ]; then
    echo "Usage: ./run_multinode.sh <node_numbers_comma_separated> <command>"
    echo "Example: ./run_multinode.sh 2,3,4,5 main_jit.py exps=ocr_noise model=JiT-E-nb"
    exit 1
fi

NODE_LIST=$1
shift
COMMAND="$@"

# Parse node numbers into array
IFS=',' read -ra NODES <<< "$NODE_LIST"
NUM_NODES=${#NODES[@]}

# First node is the master
MASTER_NODE=${NODES[0]}
MASTER_ADDR="10.142.1.${MASTER_NODE}"
MASTER_PORT=29500

echo "Master address: $MASTER_ADDR"
echo "Number of nodes: $NUM_NODES"
echo "Command: $COMMAND"

for i in "${!NODES[@]}"; do
    NODE_NUM=${NODES[$i]}
    HOSTNAME="${NODE_NUM}"
    NODE_RANK=$i
    
    echo "Starting on $HOSTNAME (node_rank=$NODE_RANK)..."
    
    ssh $HOSTNAME "bash -c '
        # 1. Change Directory FIRST so everything happens in the right place
        cd /sharedfs/mihir/continuous_diffusion/JiT || exit

        # 2. Now create logs directory inside the project folder
        mkdir -p logs
        
        # 3. Find latest log based on the project folder logs
        LATEST=\$(ls logs/*.log 2>/dev/null | sed \"s/logs\\/\\([0-9]*\\)\\.log/\\1/\" | sort -n | tail -1)
        
        if [ -z \"\$LATEST\" ]; then
            NEXT=0
        else
            NEXT=\$((LATEST + 1))
        fi
        
        LOG_FILE=\"logs/\${NEXT}.log\"
        
        MAMBA_BIN=\"/sharedfs/mihir/bin/micromamba\"
        ENV_PATH=\"/sharedfs/mihir/micromamba/envs/rae\"
        
        export PATH=\"/sharedfs/mihir/micromamba/bin:/sharedfs/mihir/bin:\$PATH\"
        export HF_HOME=/sharedfs/mihir/huggingface
        export PHO_DATA=/sharedfs/mihir
        export PHO_CHECKPOINTS=/sharedfs/mihir/pho_ckpt
        export DATA_DIR=/sharedfs/mihir/dataset
        export CKPT_DIR=/sharedfs/mihir/
        export JIT_OUT_DIR=/sharedfs/mihir/jit_ckpt    

        # 4. Run command with torchrun
        TORCHRUN_CMD=\"torchrun --nnodes=$NUM_NODES --nproc-per-node=1 --node-rank=$NODE_RANK --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT $COMMAND\"
        
        nohup \$MAMBA_BIN run -p \$ENV_PATH \$TORCHRUN_CMD > \$LOG_FILE 2>&1 &
        
        echo \"Log file: \$(pwd)/\$LOG_FILE\"
    '" &
    
    echo "Command started on $HOSTNAME"
done

wait
echo "All nodes started."