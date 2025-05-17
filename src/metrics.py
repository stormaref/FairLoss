import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss


def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def compute_ece(probs, labels, n_bins=15):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(labels)
    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        idx = np.where((conf > lo) & (conf <= hi))[0]
        if idx.size:
            acc = np.mean(preds[idx] == labels[idx])
            avg_conf = np.mean(conf[idx])
            ece += np.abs(acc - avg_conf) * (idx.size / N)
    return ece


def brier_score(probs, labels):
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


def reliability_diagram(probs, labels, n_bins=15, save_path=None):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    accs, confs = [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        idx = np.where((np.max(probs, 1) > lo) & (np.max(probs, 1) <= hi))[0]
        if idx.size:
            accs.append(np.mean(np.argmax(probs[idx], 1) == labels[idx]))
            confs.append(np.mean(np.max(probs[idx], 1)))
        else:
            accs.append(0)
            confs.append((lo + hi) / 2)

    plt.figure(figsize=(5, 5))
    plt.plot(confs, accs, marker='o')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def evaluate_metrics(probs, labels, n_bins=15, plot_path=None):
    """
    Args:
        probs (np.ndarray): Shape (N, C), softmax probabilities.
        labels (np.ndarray): Shape (N,), true labels.
    Returns:
        dict: accuracy, nll, ece, brier
    """
    metrics = {
        'accuracy': accuracy_score(labels, np.argmax(probs, axis=1)),
        'nll': log_loss(labels, probs),
        'ece': compute_ece(probs, labels, n_bins=n_bins),
        'brier': brier_score(probs, labels)
    }

    reliability_diagram(probs, labels, n_bins=n_bins, save_path=plot_path)
    return metrics
