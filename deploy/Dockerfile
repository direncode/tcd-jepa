FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install latent-ocean-core
COPY latent-ocean-core/ /tmp/latent-ocean-core/
RUN pip install /tmp/latent-ocean-core[gpu] && rm -rf /tmp/latent-ocean-core/

# Install tcd-jepa
COPY . /workspace/tcd-jepa/
RUN pip install -e /workspace/tcd-jepa/

# Default: training entrypoint
CMD ["python", "tcd-jepa/train.py", "--config", "tcd-jepa/configs/h100_ocean.yaml", "--tcd", "--backend", "ocean"]
