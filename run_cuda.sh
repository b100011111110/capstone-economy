#!/usr/bin/env bash
set -e

# Detect host driver libraries
DRIVER_LIBS=()
for lib in libcuda.so.1 libcuda.so.* libnvidia-ptxjitcompiler.so.1 libnvidia-ptxjitcompiler.so.* libnvidia-ml.so.1 libnvidia-ml.so.*; do
  for host_file in /usr/lib/x86_64-linux-gnu/$lib; do
    if [ -e "$host_file" ]; then
      DRIVER_LIBS+=("-v" "$host_file:$host_file:ro")
    fi
  done
done

# Pass NVIDIA character devices
NVIDIA_DEVICES=()
for dev in /dev/nvidia*; do
  if [ -e "$dev" ]; then
    NVIDIA_DEVICES+=("--device" "$dev")
  fi
done

# Add nvidia-smi if available
SMI_ARGS=()
if [ -x "/usr/bin/nvidia-smi" ]; then
  SMI_ARGS=("-v" "/usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro")
fi

# Run docker container with CUDA GPU acceleration
docker run --rm \
  "${NVIDIA_DEVICES[@]}" \
  "${DRIVER_LIBS[@]}" \
  "${SMI_ARGS[@]}" \
  -v "$PWD":/workspace \
  capstone-economy-cuda \
  "$@"

