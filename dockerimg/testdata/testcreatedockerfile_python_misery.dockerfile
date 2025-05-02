FROM ghcr.io/boldsoftware/sketch:86ef7a672f85139e73f38d4cdf78d95f

ARG GIT_USER_EMAIL
ARG GIT_USER_NAME

RUN git config --global user.email "$GIT_USER_EMAIL" && \
    git config --global user.name "$GIT_USER_NAME"

LABEL sketch_context="6a2899d2fa46e4791d008adc7847bb0f374fd82e23ecc9ab7a1f5f188bf7fff5"
COPY . /app

WORKDIR /app
RUN if [ -f go.mod ]; then go mod download; fi

# Install Python 3.11 (if not present in base image)
RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip || true

# Install DVC tool as requested in README
RUN pip install dvc || true

# Create and activate Python virtual environment
RUN python3.11 -m venv /app/.venv || true
ENV PATH="/app/.venv/bin:$PATH"

CMD ["/bin/sketch"]
