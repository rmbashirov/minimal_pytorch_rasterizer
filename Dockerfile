FROM nvidia/cudagl:10.2-devel-ubuntu18.04

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    build-essential \
    rsync \
    curl \
    wget \
    htop \
    git \
    openssh-server \
    nano \
    cmake \
    unzip \
    zip \
    python-opencv \
    vim \
    ffmpeg \
    tmux \
    freeglut3-dev

# cudnn
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=7.6.5.32-1+cuda10.2 \
    libcudnn7-dev=7.6.5.32-1+cuda10.2 \
    && apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# nvdiffrast setup
RUN apt-get update && apt-get install -y \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
ENV PYOPENGL_PLATFORM egl
ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH
# nvdiffrast python package is installed from requirements.txt then


RUN echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' | \
    tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json

## glew installation from source
RUN curl -L https://downloads.sourceforge.net/project/glew/glew/2.1.0/glew-2.1.0.tgz > /tmp/glew-2.1.0.tgz
RUN mkdir -p /tmp && \
    cd /tmp && tar zxf /tmp/glew-2.1.0.tgz && cd glew-2.1.0 && \
    SYSTEM=linux-egl make && \
    SYSTEM=linux-egl make install && \
    rm -rf /tmp/glew-2.1.0.zip /tmp/glew-2.1.0


# fixuid
ARG USERNAME=docker
RUN apt-get update && apt-get install -y sudo curl && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    USER=$USERNAME && \
    GROUP=$USERNAME && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml


# conda
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
# this version of miniconda's /opt/conda/bin provides pip = pip3 = pip3.7, python = python3 = python3.7
ENV PATH /opt/conda/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN pip install --upgrade pip


# python pkgs
RUN conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=10.2 -c pytorch
COPY requirements.txt /opt/requirements.txt
RUN pip --no-cache-dir install -r /opt/requirements.txt

COPY ./ /opt/minimal_pytorch_rasterizer
RUN cd /opt/minimal_pytorch_rasterizer && ./setup.sh

USER $USERNAME:$USERNAME
ENTRYPOINT ["fixuid", "-q"]
CMD ["fixuid", "-q", "bash"]
WORKDIR /src
