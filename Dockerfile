# # ===== EventSep Docker (PyTorch 2.3.0 + CUDA 12.1, RTX 3060/5080 공용) =====
# FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# # ---- 기본 유틸: FFT/오디오/컴파일러 - 확장 버전 ----
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ffmpeg git ca-certificates curl build-essential pkg-config \
#     libavcodec-dev libavformat-dev libavdevice-dev \
#     libavfilter-dev libavutil-dev libswscale-dev && \
#     rm -rf /var/lib/apt/lists/*

# # ---- Conda setup ----
# ENV CONDA_DIR=/opt/conda
# ENV PATH=$CONDA_DIR/bin:$PATH

# # ---- 전체 프로젝트 복사 (EventSep/) ----
# WORKDIR /workspace
# COPY . /workspace

# # AudioSep 레포의 실제 루트 디렉토리로 이동
# WORKDIR /workspace/AudioSep

# # ---- 이후 RUN 명령은 bash로 실행 ----
# SHELL ["/bin/bash", "-lc"]

# # ---- 새 conda env 생성 (PyTorch는 base 사용) ----
# RUN conda create -y -n EventSep python=3.10 && conda clean -ay

# # 기본 셸에서 env가 활성화되도록 설정
# RUN echo "conda activate EventSep" >> ~/.bashrc
# ENV PATH=$CONDA_DIR/envs/EventSep/bin:$PATH

# # ---- pip requirements 생성 (torch 관련 항목 제거) ----
# RUN awk '/^[[:space:]]*- pip:/{flag=1;next} \
#           flag && /^[[:space:]]*-/ {sub(/^[[:space:]]*-[[:space:]]*/, ""); print}' \
#           environment.yml \
#           | grep -v -i "^torch" \
#           | grep -v -i "pytorch" \
#           > /workspace/AudioSep/requirements_audiosep.txt

# # ---- av 10.0.0 → 최신 버전으로 교체 ----
# RUN sed -i 's/av==10.0.0/av==12.0.0/' /workspace/AudioSep/requirements_audiosep.txt

# # ---- pip 패키지 설치 ----
# RUN pip install --no-cache-dir -r requirements_audiosep.txt

# # ---- 기본 진입점 ----
# CMD ["/bin/bash"]



# ===== EventSep Docker (PyTorch 2.3.0 + CUDA 12.1) =====
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# ---- 기본 유틸 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git ca-certificates curl build-essential pkg-config \
    libavcodec-dev libavformat-dev libavdevice-dev \
    libavfilter-dev libavutil-dev libswscale-dev && \
    rm -rf /var/lib/apt/lists/*

# ---- Conda ----
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# ---- 프로젝트 ----
WORKDIR /workspace
COPY . /workspace

SHELL ["/bin/bash", "-lc"]

# ---- conda env: python만 설치 ----
RUN conda create -y -n EventSep python=3.10 pip && conda clean -ay
RUN echo "conda activate EventSep" >> ~/.bashrc
ENV PATH=$CONDA_DIR/envs/EventSep/bin:$PATH

# ---- environment.yml에서 pip 리스트만 추출 → requirements.txt 생성 ----
RUN awk '/^[[:space:]]*- pip:/{flag=1;next} \
          flag && /^[[:space:]]*-/ {sub(/^[[:space:]]*-[[:space:]]*/, ""); print}' \
          /workspace/environment.yml \
        > /workspace/requirements.txt

# ---- torch/pytorch 계열 제거 (base 이미지 사용) ----
RUN sed -i '/torch/d' /workspace/requirements.txt

# ---- av 버전 패치 ----
RUN sed -i 's/av==10.0.0/av==12.0.0/' /workspace/requirements.txt

# ---- pip 설치 ----
RUN pip install --no-cache-dir -r /workspace/requirements.txt

CMD ["/bin/bash"]
