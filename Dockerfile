FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS base

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y vim tmux zip unzip wget git build-essential libibverbs-dev ca-certificates \
    curl iproute2 libsm6 libxext6 kmod ccache libnuma-dev libssl-dev flex bison libgtk-3-dev libpango1.0-dev \
    libsoup2.4-dev libnice-dev libopus-dev libvpx-dev libx264-dev libsrtp2-dev libglib2.0-dev libdrm-dev\
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir packaging ninja cmake scikit-build-core uv meson ruff pre-commit fastapi uvicorn requests -U

RUN git clone https://github.com/vllm-project/vllm.git && cd vllm \
    && python use_existing_torch.py && pip install -r requirements/build.txt \
    && pip install --no-cache-dir --no-build-isolation -v -e .

RUN git clone https://github.com/sgl-project/sglang.git && cd sglang/sgl-kernel \
    && make build && make clean

RUN pip install --no-cache-dir diffusers transformers tokenizers accelerate safetensors opencv-python numpy imageio \
    imageio-ffmpeg einops loguru qtorch ftfy easydict

RUN conda install conda-forge::ffmpeg=8.0.0 -y && ln -s /opt/conda/bin/ffmpeg /usr/bin/ffmpeg

RUN git clone https://github.com/Dao-AILab/flash-attention.git --recursive

RUN cd flash-attention && python setup.py install && rm -rf build

RUN cd flash-attention/hopper && python setup.py install && rm -rf build

RUN git clone https://github.com/ModelTC/SageAttention.git

RUN cd SageAttention && CUDA_ARCHITECTURES="8.0,8.6,8.9,9.0,12.0" EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 pip install --no-cache-dir -v -e .

RUN git clone https://github.com/KONAKONA666/q8_kernels.git

RUN cd q8_kernels && git submodule init && git submodule update && python setup.py install && rm -rf build

# cloud deploy
RUN pip install --no-cache-dir aio-pika asyncpg>=0.27.0 aioboto3>=12.0.0 PyJWT alibabacloud_dypnsapi20170525==1.2.2 redis==6.4.0 tos -U

RUN cd /opt \
    && wget https://mirrors.tuna.tsinghua.edu.cn/gnu//libiconv/libiconv-1.15.tar.gz \
    && tar zxvf libiconv-1.15.tar.gz \
    && cd libiconv-1.15 \
    && ./configure \
    && make \
    && make install

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH=/root/.cargo/bin:$PATH

RUN cd /opt \
    && git clone https://github.com/GStreamer/gstreamer.git -b 1.24.12 --depth 1 \
    && cd gstreamer \
    && meson setup builddir \
    && meson compile -C builddir \
    && meson install -C builddir \
    && ldconfig

RUN cd /opt \
    && git clone https://github.com/GStreamer/gst-plugins-rs.git -b gstreamer-1.24.12 --depth 1 \
    && cd gst-plugins-rs \
    && cargo build --package gst-plugin-webrtchttp --release \
    && install -m 644 target/release/libgstwebrtchttp.so $(pkg-config --variable=pluginsdir gstreamer-1.0)/

RUN ldconfig

WORKDIR /workspace
