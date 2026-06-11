# Toolchain image for FPGA Event-Based Drone Collision Avoidance.
# The repo is bind-mounted at /repo (see `make docker-test`), so this image
# only carries dependencies and rebuilds only when this file or
# requirements.txt change.
FROM python:3.11-slim-trixie

# g++-13 pinned: trixie's default g++ is 14, which rejects the template
# bodies in fpga/hls_compat.h. The Makefiles invoke plain `g++`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++-13 gcc-13 make libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/g++-13 /usr/local/bin/g++ \
    && ln -s /usr/bin/gcc-13 /usr/local/bin/gcc \
    && ln -s /usr/bin/g++-13 /usr/local/bin/c++ \
    && ln -s /usr/bin/gcc-13 /usr/local/bin/cc

# CPU-only torch (no CUDA wheels) keeps the image ~1.5 GB smaller.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt flake8 clang-format==18.1.8

WORKDIR /repo
CMD ["make", "test"]
