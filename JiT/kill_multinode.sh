#!/bin/bash

# Usage: ./kill_multinode.sh 2,3,4,5
# Kills torchrun/python processes on specified nodes

if [ $# -lt 1 ]; then
    echo "Usage: ./kill_multinode.sh <node_numbers_comma_separated>"
    echo "Example: ./kill_multinode.sh 2,3,4,5"
    exit 1
fi

NODE_LIST=$1

# Parse node numbers into array
IFS=',' read -ra NODES <<< "$NODE_LIST"

echo "Killing jobs on nodes: $NODE_LIST"

for NODE_NUM in "${NODES[@]}"; do
    HOSTNAME="${NODE_NUM}"
    
    echo "Killing processes on $HOSTNAME..."
    
    ssh $HOSTNAME "bash -c '
        # Find and kill processes using GPUs
        GPU_PIDS=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u)
        if [ -n \"\$GPU_PIDS\" ]; then
            echo \"Found GPU processes: \$GPU_PIDS\"
            for PID in \$GPU_PIDS; do
                kill -9 \$PID 2>/dev/null && echo \"Killed PID \$PID\"
            done
        fi
        
        # Also kill torchrun and python processes related to training as fallback
        pkill -9 -f torchrun 2>/dev/null
        pkill -9 -f \"python.*main_jit\" 2>/dev/null
        
        echo \"Processes killed on \$(hostname)\"
    '" &
    
done

wait
echo "All jobs killed."