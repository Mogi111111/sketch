all: install check

install:
	npm ci
	npm run playwright-install

demo:
	npm run demo

# TypeScript type checking
# Note: The actual esbuild bundling happens in esbuild.go
check:
	npm test

clean:
	rm -rf node_modules
