#!/bin/bash

# ... (Previous check for arguments code remains the same) ...
if [ $# -lt 2 ]; then
    echo "Usage: ./run_cluster.sh <hostname> <whole command>"
    exit 1
fi

HOSTNAME=$1
shift
COMMAND="$@"

ssh $HOSTNAME "bash -c '
    # 1. Change Directory FIRST so everything happens in the right place
    cd /sharedfs/mihir/continuous_diffusion || exit

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
    
    export PATH="/sharedfs/mihir/micromamba/bin:/sharedfs/mihir/bin:$PATH"
    export HF_HOME=/sharedfs/mihir/huggingface
    export PHO_DATA=/sharedfs/mihir
    export PHO_CHECKPOINTS=/sharedfs/mihir/pho_ckpt
    export DATA_DIR=/sharedfs/mihir/dataset
    export CKPT_DIR=/sharedfs/mihir/
    export JIT_OUT_DIR=/sharedfs/mihir/jit_ckpt    

    # 4. Run command
    # We use -p because we are providing a full path to the env, not just a name
    nohup \$MAMBA_BIN run -p \$ENV_PATH $COMMAND > \$LOG_FILE 2>&1 &
    
    echo \"Log file: \$(pwd)/\$LOG_FILE\"
'"

echo "Command '$COMMAND' started in background on $HOSTNAME"