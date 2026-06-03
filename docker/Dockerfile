# OpenFold3 fine-tuning kit image.
# Base = the official OpenFold3 image (has run_openfold + CUDA stack installed).
# We add this kit's scripts/configs and clone the OpenFold3 *source* so that
# prepare_data.sh can reach scripts/data_preprocessing/*.py.
#
# Build (from repo root):
#   docker build -t openfold3-finetune-kit -f docker/Dockerfile .
# Run (needs an NVIDIA GPU + nvidia-container-toolkit):
#   docker run --rm --gpus all \
#     -v of3-weights:/weights -v "$PWD/work":/work \
#     openfold3-finetune-kit verify
FROM openfoldconsortium/openfold3:stable

LABEL org.opencontainers.image.title="openfold3-finetune-kit" \
      org.opencontainers.image.source="https://github.com/recep2244/openfold3-finetune-kit" \
      org.opencontainers.image.authors="Recep Adiyaman (https://github.com/recep2244)" \
      org.opencontainers.image.licenses="Apache-2.0"

# OpenFold3 source (for the data-preprocessing scripts prepare_data.sh calls).
# Pinned to a shallow clone; override OF3_REPO at runtime if you mount your own.
RUN git clone --depth 1 https://github.com/aqlaboratory/openfold-3.git /opt/openfold-3 || \
    echo "warning: could not clone openfold-3 source; mount it at /opt/openfold-3 for data prep"

WORKDIR /kit
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY examples/ ./examples/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x ./scripts/*.sh /usr/local/bin/entrypoint.sh

# Weights + work dir are volumes (weights are ~2 GB and NOT baked into the image).
ENV OPENFOLD_CACHE=/weights \
    OF3_REPO=/opt/openfold-3 \
    WORK=/work \
    CKPT=/weights/of3-p2-155k.pt
VOLUME ["/weights", "/work"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["verify"]
