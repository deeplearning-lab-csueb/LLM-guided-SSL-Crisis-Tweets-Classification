# === Stage 1: Clone the repo ===
FROM alpine/git:2.45.2 AS cloner

ARG REPO_COMMIT=main
WORKDIR /src

# Clone the repo and checkout the commit (or branch)
RUN git clone https://github.com/deeplearning-lab-csueb/LLM-guided-SSL-Crisis-Tweets-Classification.git . && \
    git checkout $REPO_COMMIT


# === Stage 2: Final runtime image ===
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /workspace

# Install vim for quick fixes
RUN apt update && apt-get install -y vim

# Copy code from the previous stage
COPY --from=cloner /src /workspace/ssl

# Create directories for shared data/artifacts
RUN mkdir -p /workspace/ssl/artifacts /workspace/ssl/data

# Install dependencies
RUN pip install --no-cache-dir -r /workspace/ssl/requirements.txt

# === W&B setup (optional — set at runtime via docker run -e WANDB_API_KEY=...) ===
ENV WANDB_API_KEY=

CMD ["bash"]
