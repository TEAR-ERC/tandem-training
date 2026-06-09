#!/bin/bash
# Clean up after a manually stopped tandem/MPI run

echo "Cleaning up MPI shared memory segments..."
rm -f /dev/shm/vader_segment.*
rm -f /dev/shm/ompi.*

echo "Cleaning up tandem checkpoint temp files..."
# Find and clean temp dirs relative to current location
find . -name "temp" -type d -exec rm -rf {}/* \; 2>/dev/null

echo "Checking for leftover MPI processes..."
LEFTOVER=$(ps aux | grep -E "tandem|mpiexec|mpirun" | grep -v grep | grep -v defunct)
if [ -n "$LEFTOVER" ]; then
    echo "Killing leftover processes..."
    pkill -9 -f tandem 2>/dev/null
    pkill -9 -f mpiexec 2>/dev/null
else
    echo "No live MPI processes found."
fi

echo "Done. /dev/shm contents:"
ls /dev/shm/ 2>/dev/null || echo "(empty)"