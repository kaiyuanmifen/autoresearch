# Container image for the autoresearch_MLE agent.
#
# This image is designed to be launched by the MLE-Bench harness:
#
#   mlebench run-agent --agent autoresearch_mle ...
#
# It mounts /home/data, /home/submission, /home/code, /home/logs at runtime,
# so the image itself contains only:
#   - a Python ML stack (numpy, pandas, scikit-learn, lightgbm, xgboost,
#     torch CPU; the harness adds GPU runtime if requested),
#   - the autoresearch_MLE agent package,
#   - an entrypoint script that boots the experiment loop.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        git \
        build-essential \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/autoresearch_mle

COPY pyproject.toml ./
COPY agent ./agent
COPY entrypoint.sh ./
COPY program.md ./

RUN pip install --upgrade pip \
    && pip install \
        numpy \
        pandas \
        scikit-learn \
        lightgbm \
        xgboost \
        scipy \
        joblib

RUN chmod +x /opt/autoresearch_mle/entrypoint.sh

# MLE-Bench mounts these directories; create them so the entrypoint never
# fails on a fresh container before the mounts land.
RUN mkdir -p /home/data /home/submission /home/code /home/logs

ENV PYTHONPATH=/opt/autoresearch_mle

ENTRYPOINT ["/opt/autoresearch_mle/entrypoint.sh"]
