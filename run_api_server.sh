#!/bin/bash
# Start the TRUST-MCNet Enhanced API Server

set -e

# Default values
PORT=8081
CONFIG_PATH="config/trust.yaml"
DB_PATH="trust_mcnet.db"
HOST="0.0.0.0"

# Help message
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -p, --port PORT        Port to run API server on (default: 8081)"
    echo "  -c, --config CONFIG    Path to config file (default: config/trust.yaml)"
    echo "  -d, --db-path PATH     Path to database file (default: trust_mcnet.db)"
    echo "  -h, --host HOST        Host address to bind to (default: 0.0.0.0)"
    echo "  --help                 Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -p|--port)
            PORT="$2"
            shift
            shift
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift
            shift
            ;;
        -d|--db-path)
            DB_PATH="$2"
            shift
            shift
            ;;
        -h|--host)
            HOST="$2"
            shift
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file '$CONFIG_PATH' not found."
    exit 1
fi

# Print startup message
echo "Starting TRUST-MCNet Enhanced API Server"
echo "========================================"
echo "Configuration:"
echo "  - Host: $HOST"
echo "  - Port: $PORT"
echo "  - Config: $CONFIG_PATH"
echo "  - Database: $DB_PATH"
echo "========================================"
echo ""

# Run the server
python examples/enhanced_api_server.py \
    --host "$HOST" \
    --port "$PORT" \
    --config "$CONFIG_PATH" \
    --db-path "$DB_PATH"
