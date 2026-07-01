#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# Ascend chip TFLOPS and bandwidth benchmark automation script.
#
# Usage:
#   ./benchmark.sh compute M N K [PLATFORM]
#   ./benchmark.sh bandwidth TOTAL_BYTES [PLATFORM]
#
# Examples:
#   ./benchmark.sh compute 4096 4096 4096
#   ./benchmark.sh bandwidth 67108864 dav-3510

set -euo pipefail

show_help() {
    echo "Ascend chip performance benchmark script"
    echo ""
    echo "Usage:"
    echo "  $0 compute M N K [PLATFORM]"
    echo "  $0 bandwidth TOTAL_BYTES [PLATFORM]"
    echo ""
    echo "Parameters:"
    echo "  PLATFORM    NPU architecture: dav-2201 (Atlas A2/A3, default) or dav-3510 (Ascend 950PR/950DT)"
    echo ""
    echo "Examples:"
    echo "  $0 compute 4096 4096 4096"
    echo "  $0 bandwidth 67108864 dav-3510"
}

if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

MODE=$1
PLATFORM="dav-2201"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p build
cd build

if [ "$MODE" = "compute" ]; then
    if [ $# -lt 4 ]; then
        echo "Error: compute mode requires M N K arguments."
        show_help
        exit 1
    fi
    M=$2
    N=$3
    K=$4
    if [ $# -ge 5 ] && [[ "$5" =~ ^dav- ]]; then
        PLATFORM=$5
    fi

    echo "Building compute benchmark for platform $PLATFORM ..."
    cmake -DCMAKE_ASC_ARCHITECTURES="$PLATFORM" ..
    make -j asc_perf_benchmark

    echo "Running compute benchmark: shape [$M, $N, $K] ..."
    ./asc_perf_benchmark compute "$M" "$N" "$K"

elif [ "$MODE" = "bandwidth" ]; then
    if [ $# -lt 2 ]; then
        echo "Error: bandwidth mode requires TOTAL_BYTES argument."
        show_help
        exit 1
    fi
    BYTES=$2
    if [ $# -ge 3 ] && [[ "$3" =~ ^dav- ]]; then
        PLATFORM=$3
    fi

    echo "Building bandwidth benchmark for platform $PLATFORM ..."
    cmake -DCMAKE_ASC_ARCHITECTURES="$PLATFORM" ..
    make -j asc_perf_benchmark

    echo "Running bandwidth benchmark: $BYTES bytes ..."
    ./asc_perf_benchmark bandwidth "$BYTES"

else
    echo "Error: unknown mode '$MODE'. Use 'compute' or 'bandwidth'."
    show_help
    exit 1
fi

echo ""
echo "Benchmark complete."
