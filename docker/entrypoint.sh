#!/bin/bash
# LLMSQL2 Docker Startup Script

echo "============================================"
echo "LLMSQL2 - Text-to-SQL Docker Application"
echo "============================================"

# Check if running in development mode
if [ "$DEV_MODE" = "true" ]; then
    echo "Running in development mode..."
    exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
fi

# Check for specific command
case "$1" in
    jupyter)
        echo "Starting Jupyter Lab..."
        exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
        ;;
    notebook)
        echo "Starting Jupyter Notebook..."
        exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
        ;;
    python)
        shift
        echo "Running Python script: $@"
        exec python "$@"
        ;;
    shell)
        echo "Starting shell..."
        exec /bin/bash
        ;;
    *)
        # Default: Start Jupyter Lab
        echo "Starting Jupyter Lab (default)..."
        exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
        ;;
esac
