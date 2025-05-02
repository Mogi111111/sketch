#!/usr/bin/env bash

# This script lists all currently running sketch containers in Docker,
# displaying their container names, local URLs, and sketch titles.
# It extracts port mappings and queries each sketch's state endpoint
# to provide a convenient overview of running sketches.

docker ps --format "{{.Names}}|{{.Ports}}" | \
  grep sketch | \
  sed -E 's/.*0\.0\.0\.0:([0-9]+)->80.*/\1/' | \
  while read port; do
    # Get container name for this port
    name=$(docker ps --filter "publish=$port" --format "{{.Names}}")
    
    # Get sketch title from its state endpoint
    title=$(curl -s "http://localhost:$port/state" | jq -r '.title // "N/A"')
    
    # Format and print the result
    printf "%-30s http://localhost:%d/   %s\n" "$name" "$port" "$title"
  done
