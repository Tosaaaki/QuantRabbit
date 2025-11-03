#!/usr/bin/env bash
# Install Google Cloud SDK (gcloud) on macOS or Debian/Ubuntu.
# Safe to re-run. Requires sudo on Linux.

set -euo pipefail

echo "[install_gcloud] Detecting platform..."
OS="$(uname -s)"

has() { command -v "$1" >/dev/null 2>&1; }

if has gcloud; then
  echo "[install_gcloud] gcloud already installed: $(gcloud version | head -n1)"
  exit 0
fi

case "$OS" in
  Darwin)
    if has brew; then
      echo "[install_gcloud] Installing via Homebrew (google-cloud-sdk)..."
      brew install --cask google-cloud-sdk || brew install google-cloud-sdk
      echo "[install_gcloud] Installed. Initialize with: gcloud init"
    else
      echo "[install_gcloud] Homebrew 未検出。公式アーカイブから導入します..."
      tmpdir="$(mktemp -d)"; trap 'rm -rf "$tmpdir"' EXIT
      cd "$tmpdir"
      arch_tag="$(uname -m)"
      case "$arch_tag" in
        x86_64) arch_tag="x86_64" ;;
        arm64|aarch64) arch_tag="arm" ;;
        *) echo "[install_gcloud] 未対応アーキテクチャ: $(uname -m)" >&2; exit 2 ;;
      esac
      url="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-darwin-${arch_tag}.tar.gz"
      echo "[install_gcloud] Downloading: $url"
      curl -fsSLO "$url"
      tarball=$(basename "$url")
      tar xf "$tarball"
      ./google-cloud-sdk/install.sh -q
      echo "export PATH=\"$HOME/google-cloud-sdk/bin:$PATH\"" >> "$HOME/.bashrc"
      echo "export PATH=\"$HOME/google-cloud-sdk/bin:$PATH\"" >> "$HOME/.zshrc"
      echo "[install_gcloud] Installed. Restart shell and run: gcloud init"
    fi
    ;;
  Linux)
    # Prefer Debian/Ubuntu apt installation. Fallback to official tar if apt is unavailable.
    if has apt-get; then
      echo "[install_gcloud] Installing via apt (requires sudo)..."
      sudo apt-get update -y
      sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
      echo "[install_gcloud] Adding Google Cloud apt repository..."
      curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
      echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
      sudo apt-get update -y
      sudo apt-get install -y google-cloud-cli
      echo "[install_gcloud] Installed. Initialize with: gcloud init"
    else
      echo "[install_gcloud] apt 未検出。公式インストーラで導入します..."
      tmpdir="$(mktemp -d)"; trap 'rm -rf "$tmpdir"' EXIT
      cd "$tmpdir"
      curl -fsSLO https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-$(uname -m | sed 's/x86_64/x86_64/; s/aarch64/arm/').tar.gz || true
      # Fallback to generic archive name
      if ! ls google-cloud-cli-*.tar.gz >/dev/null 2>&1; then
        curl -fsSLO https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli.tar.gz
        tarball=google-cloud-cli.tar.gz
      else
        tarball=$(ls google-cloud-cli-*.tar.gz | head -n1)
      fi
      tar xf "$tarball"
      ./google-cloud-sdk/install.sh -q
      echo "export PATH=\"$HOME/google-cloud-sdk/bin:$PATH\"" >> "$HOME/.bashrc"
      echo "export PATH=\"$HOME/google-cloud-sdk/bin:$PATH\"" >> "$HOME/.zshrc"
      echo "[install_gcloud] Installed. Restart shell and run: gcloud init"
    fi
    ;;
  *)
    echo "[install_gcloud] Unsupported OS: $OS. Install manually: https://cloud.google.com/sdk/docs/install" >&2
    exit 2
    ;;
esac
