#!/bin/bash
# One-liner to zip checkpoint folder for download
# Usage: ./scripts/zip_checkpoint.sh

# Zip checkpoint folder
cd /hai/users/a/s/asanshay/websight-v2/saves/qwen3-vl-8b/lora/sft-noeval && \
tar -czf /hai/scratch/asanshay/websight-v2/checkpoint-200.tar.gz checkpoint-200 && \
echo "✓ Checkpoint zipped to: /hai/scratch/asanshay/websight-v2/checkpoint-200.tar.gz" && \
du -h /hai/scratch/asanshay/websight-v2/checkpoint-200.tar.gz

