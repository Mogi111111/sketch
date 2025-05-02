FROM ghcr.io/boldsoftware/sketch:86ef7a672f85139e73f38d4cdf78d95f

ARG GIT_USER_EMAIL
ARG GIT_USER_NAME

RUN git config --global user.email "$GIT_USER_EMAIL" && \
    git config --global user.name "$GIT_USER_NAME"

LABEL sketch_context="2f264f73c8a474c0901bf67b0b1ae2ffe24afc4aceabc27435efbd70ed4c36ab"
COPY . /app

WORKDIR /app
RUN if [ -f go.mod ]; then go mod download; fi

# No additional setup needed for this simple Go test project

CMD ["/bin/sketch"]
