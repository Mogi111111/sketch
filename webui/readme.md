# Loop WebUI

A modern web interface for the CodingAgent loop.

The server in the sibling directory (../server) exposes an HTTP API for
the CodingAgent.

## Development

This module contains a TypeScript-based web UI for the Loop service. The TypeScript code is compiled into JavaScript using esbuild, and the resulting bundle is served by the Go server.

### Prerequisites

- Node.js and npm
- Go 1.20 or later

### Setup

```bash
# Install dependencies
make install

# Run tests
make check
```

### Development Mode

For development, you can use watch mode:

```bash
make demo
```

This will launch a local web server that serves the demo pages for the web components. You can edit the TypeScript files, and the changes will be reflected in real-time.

#### VSCode

If you are using Visual Studio Code, you can use the `Launch Chrome against localhost` launch configuration to run the demo server. This configuration is set up to automatically open a sketch page with dummy data in Chrome when you start debugging, supporting hot module reloading and breakpoints.

## Integration with Go Server

The TypeScript code is bundled into JavaScript using esbuild and then served by the Go HTTP server. The integration happens through the `webui` package, which provides a function to retrieve the built bundle.

The server code accesses the built web UI through the `webui.GetBundle()` function, which returns a filesystem that can be used to serve the files.

## File Structure

- `src/`: TypeScript source files
- `dist/`: Generated JavaScript bundle
- `esbuild.go`: Go code for bundling TypeScript files
- `Makefile`: Build tasks

## Bundle Analysis

You can analyze the size and dependency structure of the TypeScript bundles:

```bash
# Generate bundle metafiles in a temporary directory
go run sketch.dev/cmd/bundle-analyzer
```

The tool generates metafiles that can be analyzed by dragging them onto the esbuild analyzer at https://esbuild.github.io/analyze/
