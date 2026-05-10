#!/usr/bin/env bash
set -e

export MVCAM_SDK_PATH=/opt/MVS
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export MVCAM_GENICAM_CLPROTOCOL=/opt/MVS/lib/CLProtocol
export ALLUSERSPROFILE=/opt/MVS/MVFG
export LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib/32:$LD_LIBRARY_PATH

WORK_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
BUILD_DIR="$WORK_DIR/build"
CONFIG_DIR="$WORK_DIR/config.yaml"
SH_DIR="$WORK_DIR/setup_service.sh"

ACTION="${1:-build}"   # 默认 build
echo "[INFO] Action: $ACTION"

if [[ "$ACTION" == "clean" || "$ACTION" == "rebuild" ]]; then
    echo "[INFO] Cleaning build directories..."
    rm -rf  "$BUILD_DIR"

    if [[ "$ACTION" == "clean" ]]; then
        echo "[INFO] Clean done."
        exit 0
    fi
fi

mkdir -p "$BUILD_DIR"

# cmake 配置
cmake -S "$WORK_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release

# 编译
make -C "$BUILD_DIR"

echo "[INFO] Build finished successfully."