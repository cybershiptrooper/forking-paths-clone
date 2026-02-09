# Use Runpod PyTorch image with CUDA 12.4.1
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Define environment variables for Hugging Face and W&B tokens (to be set at runtime)
# ENV HF_TOKEN=""             # Hugging Face API token (export HF_TOKEN=<token> when running)
# ENV WANDB_API_KEY=""        # W&B API key (export WANDB_API_KEY=<key> when running)

# Install system packages (tmux, git-lfs) and clean up apt caches
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        tmux \
        git-lfs \
 && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Install Python 3.12 using uv (which will be used by uv sync)
RUN uv python install 3.12

# Configure Git user name and email globally
RUN git config --global user.name "cybershiptrooper" \
 && git config --global user.email "cybershiptrooper@gmail.com"

# Set working directory
WORKDIR /workspace

# Initialize git-lfs (needed before copying files)
RUN git lfs install

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies using uv (this layer will be cached if dependencies don't change)
# uv will use Python 3.12 that we installed above
RUN uv sync --python 3.12

# Copy the rest of the project files
COPY . .

# Set up environment to use uv's virtual environment by default
ENV PATH="/workspace/.venv/bin:$PATH"

# (Optional) If you want automatic script activation on container start, 
# you could add an ENTRYPOINT or source ~/.bashrc here. For example:
# ENTRYPOINT ["/bin/bash", "-c", "source /workspace/.venv/bin/activate && exec \"$@\"", "--"]
