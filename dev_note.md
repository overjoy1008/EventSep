### docker image
docker build -t eventsep-kbp:latest .

### docker container
docker run --gpus all -it --rm \
    --name eventsep_dev \
    -v /mnt/c/Users/EP800-202H/Gradient/Github/EventSep:/workspace \
    -w /workspace/AudioSep \
    eventsep-kbp:latest \
    /bin/bash



### Notes
environment.yml은 AudioSep에서 가져다 사용하였으나 일부 버전 업데이트를 진행:
- av: 10.0.0 -> 12.0.0