#!/bin/sh
# Exit immediately if a command exits with a non-zero status
set -e

# Detect the architecture
ARCH=$(uname -m)

if [ "$ARCH" = "x86_64" ]; then
    S5CMD_ARCH="Linux-64bit"
elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    # Added "arm64" catch-all just in case of environment variances
    S5CMD_ARCH="Linux-ARM64"
else
    echo "ERROR: Unsupported architecture: $ARCH" >&2
    exit 1
fi

echo "Downloading s5cmd v2.3.0 for $S5CMD_ARCH..."

# Download and extract directly into /usr/local/bin
curl -sSL "https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_${S5CMD_ARCH}.tar.gz" | \
tar -xz -C /usr/local/bin/ s5cmd

echo "s5cmd installed successfully to /usr/local/bin/s5cmd"