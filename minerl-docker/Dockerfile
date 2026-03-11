# =============================================================================
# MineRL Docker Image
# Supports: Mac (CPU/x86 emulation) and Windows (GPU via NVIDIA runtime)
# Usage: see docker-compose.yml or README.md
# =============================================================================

ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV DISPLAY=:99
ENV PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    openjdk-8-jdk \
    python3.9 \
    python3-pip \
    python3.9-dev \
    python3.9-distutils \
    git \
    curl \
    wget \
    unzip \
    xvfb \
    x11vnc \
    xterm \
    novnc \
    websockify \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-dev \
    mesa-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Python toolchain pinned for MineRL compatibility ─────────────────────────
RUN python3.9 -m pip install --upgrade \
    "pip==23.3.1" \
    "setuptools<58.0.0" \
    "wheel==0.37.1"

# ── Install gym 0.19.0 (no-build-isolation bypasses broken metadata) ──────────
RUN python3.9 -m pip install \
    "numpy>=1.18.0" \
    "cloudpickle>=1.2.0,<1.7.0" \
    "pyglet>=1.4.0" \
    "Pillow" \
    "scipy" \
    && python3.9 -m pip install "gym==0.19.0" --no-build-isolation

# ── Clone MineRL and apply MixinGradle patch ─────────────────────────────────
WORKDIR /opt
RUN git clone https://github.com/minerllabs/minerl.git minerl_src \
    && cd minerl_src \
    && git checkout v0.4.4

# Download the MixinGradle zip (used to build from source)
RUN mkdir -p /opt/local_maven/MixinGradle-dcfaf61/MixinGradle/dcfaf61 \
    && wget -q "https://github.com/SpongePowered/MixinGradle/archive/dcfaf61.zip" \
       -O /tmp/mixin.zip \
    && echo "Attempting SpongePowered archive..." \
    || echo "SpongePowered archive failed, will use local stub"

# Patch build.gradle to use local flat-dir repo for MixinGradle
RUN GRADLE_FILE="/opt/minerl_src/minerl/Malmo/Minecraft/build.gradle" \
    && sed -i "s|jcenter()|jcenter()\n        flatDir { dirs '/opt/local_maven' }|g" "$GRADLE_FILE" \
    && sed -i "s|classpath('com.github.SpongePowered:MixinGradle:dcfaf61')|classpath('MixinGradle-dcfaf61:MixinGradle:dcfaf61')|g" "$GRADLE_FILE" \
    && sed -i "s|net.minecraftforge.gradle:ForgeGradle:2.2-SNAPSHOT|com.github.brandonhoughton:ForgeGradle:FG_2.2_patched-SNAPSHOT|g" "$GRADLE_FILE"

# Copy the MixinGradle jar into the flat-dir
COPY MixinGradle-dcfaf61.jar /opt/local_maven/MixinGradle-dcfaf61-dcfaf61.jar

# ── Build and install MineRL ──────────────────────────────────────────────────
WORKDIR /opt/minerl_src
RUN python3.9 -m pip install -e . --no-build-isolation

# ── Jupyter + ML libraries ────────────────────────────────────────────────────
RUN python3.9 -m pip install \
    jupyter \
    jupyterlab \
    ipywidgets \
    matplotlib \
    torch \
    torchvision \
    stable-baselines3 \
    tensorboard \
    opencv-python-headless \
    imageio

# ── Copy startup script ───────────────────────────────────────────────────────
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# ── Workspace ─────────────────────────────────────────────────────────────────
RUN mkdir -p /workspace
WORKDIR /workspace

COPY examples/test_minerl.ipynb /workspace/test_minerl.ipynb

EXPOSE 8888   
EXPOSE 5900   
EXPOSE 6080   

CMD ["/usr/local/bin/start.sh"]