#!/bin/bash
set -eux

# 1) OpenSSH Server を導入して起動
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openssh-server
systemctl enable --now ssh

# 2) Docker だけ最小構成で導入
apt-get install -y --no-install-recommends docker.io
systemctl enable --now docker

# 3) Bot コンテナをバックグラウンド実行
docker run -d \
  --name quantrabbit \
  asia-northeast1-docker.pkg.dev/quantrabbit/fx/quantrabbit:latest