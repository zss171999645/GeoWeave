# start docker locally
docker run --name vggt  \
    --shm-size=2g  \
    --gpus all \
    -e HOME=/home/users/$USER \
    -u $(id -u):$(id -g) \
    -v /etc/passwd:/etc/passwd:ro \
    -v /horizon-bucket:/horizon-bucket \
    -v $(pwd):/workspace \
    -v /home/users/$USER:/home/users/$USER \
    --net=host  \
    -it \
    docker.hobot.cc/imagesys/base:centos7.6-gcc11.4-py3.11-cu12.4-rdma-torch2.6.0-fa3-spattn \
    /bin/bash -c "cd /workspace && source /opt/miniconda3/etc/profile.d/conda.sh && conda activate easyvolcap && exec /bin/bash"