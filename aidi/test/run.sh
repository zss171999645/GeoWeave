# 0. Install the package and dependencies
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda"
export CUDA_DEVICE_ORDER=PCI_BUS_ID # OPTIONAL: defaults to capability order, might be different for GL and CUDA
export PATH="/opt/miniconda3/bin:$PATH"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate easyvolcap

accelerate launch \
    --config_file accelerate.yaml \
    --machine_rank=$NODE_RANK \
    --main_process_ip=$HOST_NODE_ADDR \
    --num_machines=$NUM_NODES \
    --num_processes=$WORLD_SIZE \
    --main_process_port=28767 \
    test/test.py
