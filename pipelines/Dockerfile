FROM python:3.9-slim-buster
# Install apt dependencies
RUN apt-get update && apt-get install -y \
    gpg-agent \
    python3-cairocffi \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget

WORKDIR /training

# COPY and install requirements first
COPY pipelines/training-requirements.txt ./training-requirements.txt
RUN pip install -r training-requirements.txt --extra-index-url ${PIP_EXTRA_INDEX_URL}

# COPY remaining files by order of likelihood of changing
COPY pipeline.sh ./
COPY pottery.json ./
COPY data ./data
COPY triage_bandit_sandbox ./triage_bandit_sandbox