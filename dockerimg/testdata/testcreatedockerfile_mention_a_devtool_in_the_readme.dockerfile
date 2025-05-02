FROM ghcr.io/boldsoftware/sketch:86ef7a672f85139e73f38d4cdf78d95f

ARG GIT_USER_EMAIL
ARG GIT_USER_NAME

RUN git config --global user.email "$GIT_USER_EMAIL" && \
    git config --global user.name "$GIT_USER_NAME"

LABEL sketch_context="8e7669851050cbea949f078549508f32fcd5014dc77e548c76bd9041db4e249a"
COPY . /app

WORKDIR /app
RUN if [ -f go.mod ]; then go mod download; fi

# Install graphviz for the 'dot' command needed for tests
RUN --mount=type=cache,target=/var/cache/apt \
    set -eux; \
    apt-get update && \
    apt-get install -y --no-install-recommends graphviz || true

CMD ["/bin/sketch"]
