#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_mcp.sh — launch the Transcript Intelligence MCP server via Docker
#
# Claude Desktop on Mac calls this script as the MCP server command.
# It exec's into the running container and starts the server over stdio.
#
# Usage:
#   1. Start the container:  docker start transcript-intelligence
#   2. Point Claude Desktop at this script (see README for config snippet)
#   3. Claude Desktop calls: bash scripts/run_mcp.sh
# ---------------------------------------------------------------------------

# EDIT this if your container has a different name (check: docker ps)
CONTAINER_NAME="transcript-intelligence"

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "ERROR: Container '${CONTAINER_NAME}' is not running." >&2
  echo "       Start it with:  docker start ${CONTAINER_NAME}" >&2
  exit 1
fi

exec docker exec -i "${CONTAINER_NAME}" python -m mcp_server.server
