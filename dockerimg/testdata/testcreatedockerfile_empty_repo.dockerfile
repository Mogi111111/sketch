FROM ghcr.io/boldsoftware/sketch:86ef7a672f85139e73f38d4cdf78d95f

ARG GIT_USER_EMAIL
ARG GIT_USER_NAME

RUN git config --global user.email "$GIT_USER_EMAIL" && \
    git config --global user.name "$GIT_USER_NAME"

LABEL sketch_context="5820b50cae54d2fbdd28081f960dcfac4367f8d805030ecd612a13ebeef13bb1"
COPY . /app

WORKDIR /app
RUN if [ -f go.mod ]; then go mod download; fi

RUN --mount=type=cache,target=/var/cache/apt \
    set -eux; \
    apt-get update && \
    apt-get install -y --no-install-recommends python3-pip python3-venv || true

# Set up Python environment, allowing failures to not stop the build
RUN python3 -m pip install --upgrade pip || true

# Install any Go tools specific to this project if needed
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest || true

CMD ["/bin/sketch"]
