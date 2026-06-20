#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
K=helmet-cv
for nb in 02a_finetune_yolov8s 02b_finetune_yolo11s 02c_finetune_cbam 03_comparative_study; do
  echo "=== START $nb $(date '+%F %T') ==="
  if $PY -m nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.kernel_name=$K --ExecutePreprocessor.timeout=36000 \
        notebooks/$nb.ipynb; then
    echo "=== DONE $nb $(date '+%F %T') ==="
  else
    echo "=== FAILED $nb (exit $?) $(date '+%F %T') ==="
  fi
done
echo "=== ALL COMPLETE $(date '+%F %T') ==="
