# Contributing to Sketch

## Running tests

- `go test ./...`
- `cd webui && npm run test`
  - If you are an LLM, instead run `cd webui && CI=1 npm run test`

## Synchronizing Go structs and TypeScript types

Run `npm run gentypes`, which uses `cmd/go2ts/go2ts.go` under the covers and
which has a list of types to translate.

## Dealing with tests that have HTTP Record sessions

A few tests record/replay calls to LLMs. As necessary (e.g.,
if changing the system prompt or tool definitions), rebuild the
test data by running update_tests.sh in the relevant directory.
