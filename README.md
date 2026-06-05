# FairLoss

PyTorch classification loss that combines cross-entropy with a smooth argmax penalty. Inspired by **Penalized Logarithmic Loss (PLL)** from Ahmadian et al. (2025).

## The problem

In multi-class classification, we often want losses to rank **correct** predictions above **incorrect** ones. Strictly proper scores such as cross-entropy encourage calibrated probabilities, but they do not always do that.

Consider two logits with true class `0`:

| Sample | Logits               | True class | Argmax correct? |
| ------ | -------------------- | ---------- | --------------- |
| 1      | `[0.34, 0.33, 0.33]` | 0          | Yes             |
| 2      | `[0.49, 0.51, 0.00]` | 0          | No              |

Sample 2 assigns a **higher logit to the true class** but picks the **wrong label**. Cross-entropy prefers sample 2; intuitively, sample 1 should win.

Ahmadian et al. call this the **Superior** property: a scoring rule should always rank correct predictions above incorrect ones. PLL adds a hard argmax penalty while staying strictly proper. FairLoss applies the same idea with a differentiable penalty suitable for gradient-based training.

Running [`examples/difference.py`](examples/difference.py) on these logits:

```
Cross-entropy:  tensor([1.0920, 0.9681])   # sample 2 wins
FairLoss:       tensor([1.0920, 1.3014])   # sample 1 wins
```

Cross-entropy gives sample 2 the lower loss because it is more confident on the true class. FairLoss adds a penalty when the argmax is wrong, so sample 1 ends up with the lower loss. Sample 1 keeps the same loss under both criteria because its prediction is already correct.

## How it works

FairLoss keeps standard cross-entropy and adds a SmoothMax-based penalty:

```
total_loss = cross_entropy + penalty
```

The penalty is built from the gap between the true-class logit and a smooth approximation of the largest logit:

1. **SmoothMax** approximates `max(logits)` — the predicted class.
2. **`logit_true - smooth_max`** is near zero when the argmax is correct, negative when it is wrong.
3. A **sigmoid transform** maps that gap into a smooth penalty: near zero for correct predictions, positive for incorrect ones.

Full derivation: [`formulation.pdf`](formulation.pdf)

## Install

```bash
pip install fairloss
```

Or install from source:

```bash
pip install git+https://github.com/stormaref/FairLoss.git
```

## Quick start

```python
import torch
import torch.nn as nn
import torch.optim as optim
from fairloss import FairLoss

model = nn.Linear(784, 10)
criterion = FairLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

x = torch.randn(64, 784)
y = torch.randint(0, 10, (64,))

logits = model(x)
loss = criterion(logits, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

See [`examples/basic_usage.py`](examples/basic_usage.py) for a training loop and [`examples/difference.py`](examples/difference.py) for the cross-entropy comparison above.

## API

### `FairLoss(cross_entropy=None, smooth_max_beta=1000.0, scaling_factor=1000.0, reduction="mean")`

Wraps a `nn.CrossEntropyLoss` and adds the FairLoss penalty on top.

| Parameter         | Description                                                                 |
| ----------------- | --------------------------------------------------------------------------- |
| `cross_entropy`   | Optional `nn.CrossEntropyLoss` with `reduction="none"`. Defaults to `nn.CrossEntropyLoss(reduction="none")`. Pass a custom instance to set `weight`, `label_smoothing`, `ignore_index`, etc. |
| `smooth_max_beta` | Temperature for the smooth maximum approximation over logits.               |
| `scaling_factor`  | Scale applied before the sigmoid penalty transform.                         |
| `reduction`       | Reduction applied to the combined loss: `"mean"`, `"sum"`, or `"none"`. Default: `"mean"`. |

**Forward pass**

- `output`: Logits tensor of shape `(N, C)` where `N` is batch size and `C` is number of classes.
- `target`: Integer class labels of shape `(N,)`.
- **Returns**: Scalar mean loss over the batch (or per-sample losses when `reduction="none"`).

## Requirements

- Python 3.9+
- PyTorch 2.0+

## License

MIT — see [LICENSE](LICENSE).

## Reference

Ahmadian, R., Ghatee, M., & Wahlström, J. (2025). Superior scoring rules for probabilistic evaluation of single-label multi-class classification tasks. _International Journal of Approximate Reasoning_, 182, 109421. [https://doi.org/10.1016/j.ijar.2025.109421](https://doi.org/10.1016/j.ijar.2025.109421)

Preprint: [arXiv:2407.17697](https://arxiv.org/abs/2407.17697)

GitHub: https://github.com/stormaref/FairLoss
