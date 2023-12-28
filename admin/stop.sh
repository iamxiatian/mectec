#!/bin/bash

# Send SIGTERM to all processes with "main.py" in their name
pkill -TERM -f "main.py"

# Wait for processes to terminate gracefully
sleep 5

# Send SIGKILL to any remaining processes
pkill -KILL -f "main.py"