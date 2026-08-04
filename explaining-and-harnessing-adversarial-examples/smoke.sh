#!/usr/bin/env bash
# smoke.sh — the same code path at a size that finishes in a couple of minutes.
# Proves the path runs; NOT evidence about the paper (never report its output
# as a result). Prints one FINAL line.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

EAE_SMOKE=1 EAE_ONLY=softmax_reg exec "$PY" -c "
import os, time, torch, numpy as np
import data, models, train, eval as ev
torch.set_num_threads(2)
# tiny: 2000 train examples, 5 epochs, seed 0 only
d = data.load_mnist(0)
import torch as T
dt = {k: T.from_numpy(v) for k,v in d.items()}
dt['x_train'] = dt['x_train'][:2000]; dt['y_train'] = dt['y_train'][:2000]
m = models.SoftmaxRegression()
h = train.train(m, dt, {'lr':0.5,'max_epochs':5,'batch_size':256,'seed':0,'momentum':0.9})
adv = ev.adv_eval(m, dt['x_test'][:500], dt['y_test'][:500], 0.25)
print(f'FINAL softmax_reg={adv[\"adv_err\"]:.4f}')
"
