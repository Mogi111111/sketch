FROM ghcr.io/boldsoftware/sketch:86ef7a672f85139e73f38d4cdf78d95f

ARG GIT_USER_EMAIL
ARG GIT_USER_NAME

RUN git config --global user.email "$GIT_USER_EMAIL" && \
    git config --global user.name "$GIT_USER_NAME"

LABEL sketch_context="c4ea887ac5976c2017ccdc471f243212da6432801db50cac2c94977288af310f"
COPY . /app

WORKDIR /app
RUN if [ -f go.mod ]; then go mod download; fi

# Install Node.js related tools with corepack
RUN npm install -g corepack && corepack enable || true

# Ensure go tests can run
RUN go install gotest.tools/gotestsum@latest || true

CMD ["/bin/sketch"]
