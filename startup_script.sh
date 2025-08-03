#!/bin/bash
set -ex

echo "Starting QuantRabbit VM setup script (Minimal for SSH debug)..."

# Create tossaki user and grant sudo privileges
echo "Creating tossaki user and granting sudo privileges..."
useradd -m -s /bin/bash tossaki
usermod -aG sudo tossaki
echo "tossaki ALL=(ALL) NOPASSWD: ALL" | tee /etc/sudoers.d/tossaki-nopasswd > /dev/null

echo "QuantRabbit VM setup script (Minimal) finished."