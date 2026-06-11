#!/bin/bash

# 1. Dynamically query the Linux kernel for all allowed CPU cores.
# Python's sched_getaffinity returns exactly the allowed cores as a sorted list.
CORES=($(python3 -c "import os; print(' '.join(map(str, sorted(os.sched_getaffinity(0)))))"))

# Count how many cores we actually have
NUM_CORES=${#CORES[@]}

# 2. Get this specific process's MPI rank
RANK=$OMPI_COMM_WORLD_RANK

# 3. Assign a specific core from the array based on the rank.
# Using modulo (%) makes this bulletproof: if you accidentally run 40 ranks on 30 cores,
# it will safely loop back around and double up, rather than crashing out of bounds.
CORE_INDEX=$((RANK % NUM_CORES))
MY_CORE=${CORES[$CORE_INDEX]}

# 4. Execute Tandem pinned strictly to that single dynamic core
exec taskset -c $MY_CORE "$@"