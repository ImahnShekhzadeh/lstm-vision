#!/bin/sh
isort /app/scripts/*.py
black /app/scripts/*.py

python3 -B /app/scripts/run.py --bidirectional --num_epochs 1 --seed_number 0 --tex__file_path "out/out.tex" --use_amp