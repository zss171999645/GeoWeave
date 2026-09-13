# start docker locally
docker run --name vggt_root  \
    --gpus all \
    -u 0:0 \
    -v /etc/passwd:/etc/passwd:ro \
    -v /horizon-bucket:/horizon-bucket \
    -v $(pwd):/workspace \
    --net=host  \
    -e HTTP_PROXY= \
    -e HTTPS_PROXY= \
    -e http_proxy= \
    -e https_proxy= \
    -it \
    docker.hobot.cc/imagesys/base:centos7.6-gcc11.4-py3.11-cu12.4-rdma-torch2.6.0-fa3-libyaml \
    /bin/bash -c "cd /workspace && source /opt/miniconda3/etc/profile.d/conda.sh && conda activate easyvolcap && exec /bin/bash"
