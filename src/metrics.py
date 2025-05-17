import numpy as np
import matplotlib.pyplot as plt

def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def compute_ece(probs, labels, n_bins=10):
    """
    probs: shape (N, C)
    labels: shape (N,)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(labels)
    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        idx = np.where((conf > lo) & (conf <= hi))[0]
        if idx.size:
            acc = np.mean(preds[idx] == labels[idx])
            avg_conf = np.mean(conf[idx])
            ece += np.abs(acc - avg_conf) * (idx.size / N)
    return ece

def brier_score(probs, labels):
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1
    return np.mean(np.sum((probs - one_hot)**2, axis=1))

def reliability_diagram(ax, probs, labels, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    accs, confs = [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        idx = np.where((np.max(probs, 1) > lo) & (np.max(probs, 1) <= hi))[0]
        if idx.size:
            accs.append(np.mean(np.argmax(probs[idx], 1) == labels[idx]))
            confs.append(np.mean(np.max(probs[idx], 1)))
        else:
            accs.append(0)
            confs.append((lo + hi) / 2)
    ax.plot(confs, accs, marker='o')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram')

def evaluate_mc_metrics(mc_logits, labels, n_bins=10):
    """
    mc_logits: np.ndarray of shape (T, N, C)
    labels: np.ndarray of shape (N,)
    """
    if isinstance(mc_logits, np.ndarray) is False:
        mc_logits = mc_logits.cpu().numpy()

    T, N, C = mc_logits.shape
    mc_probs = softmax(mc_logits)  # (T, N, C)
    mean_probs = np.mean(mc_probs, axis=0)  # (N, C)

    acc = np.mean(np.argmax(mean_probs, axis=1) == labels)
    ece = compute_ece(mean_probs, labels, n_bins=n_bins)
    brier = brier_score(mean_probs, labels)

    print(f"Accuracy:     {acc:.4f}")
    print(f"ECE:          {ece:.4f}")
    print(f"Brier Score:  {brier:.4f}")

    fig, ax = plt.subplots()
    reliability_diagram(ax, mean_probs, labels, n_bins=n_bins)
    plt.tight_layout()
    plt.show()

    return {"accuracy": acc, "ece": ece, "brier": brier}
