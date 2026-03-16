# syntax=docker/dockerfile:1.4
ARG BASE_IMAGE=python:3.13.5-slim-bookworm
FROM ${BASE_IMAGE} AS builder

# Metadata passed at build time
ARG VERSION=0.1.0
ARG GIT_HASH=unknown
ARG CREATION_DATE=unknown
ARG BASE_IMAGE_DIGEST=unspecified

ENV PYTHONDONTWRITEBYTECODE=1 \
	PIP_NO_CACHE_DIR=1 \
	DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy wheel and install
COPY dist/replicnn-${VERSION}-py3-none-any.whl ./
RUN pip install replicnn-${VERSION}-py3-none-any.whl
    
# Final slim image
FROM ${BASE_IMAGE}

ARG VERSION
ARG GIT_HASH
ARG CREATION_DATE
ARG BASE_IMAGE
ARG BASE_IMAGE_DIGEST

ENV PATH="/opt/replicnn/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /usr/local /usr/local

WORKDIR /data

ENTRYPOINT ["replicnn"]

LABEL org.opencontainers.image.authors="Dominik Stroh <dominik.stroh@uni-wuerzburg.de>, Kathi Zarnack <kathi.zarnack@uni-wuerzburg.de>" \
      org.opencontainers.image.url="https://github.com/ZarnackGroup/RepliCNN" \
      org.opencontainers.image.documentation="https://github.com/ZarnackGroup/RepliCNN" \
      org.opencontainers.image.source="https://github.com/ZarnackGroup/RepliCNN" \
      org.opencontainers.image.licenses="GPLv3" \
      org.opencontainers.image.title="RepliCNN" \
      org.opencontainers.image.description="RepliCNN is a tool for predicting replication timing from DNA sequence data using convolutional neural networks." \
      org.opencontainers.image.revision="${GIT_HASH}" \
      org.opencontainers.image.created="${CREATION_DATE}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      org.opencontainers.image.base.digest="${BASE_IMAGE_DIGEST}"