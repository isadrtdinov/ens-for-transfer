import torch

def accuracy_score(preds, labels):
    return (preds == labels).float().mean().item()

def top1_acc(logits, labels):

    return accuracy_score(logits.argmax(dim=-1), labels)

def mean_per_class_acc(logits, labels):
    result = 0.
    unique_labels = torch.unique(labels)
    preds = logits.argmax(dim=-1)
    for label in unique_labels:
        ind = (labels == label)
        score = accuracy_score(preds[ind], labels[ind])
        result += accuracy_score(preds[ind], labels[ind])

    return result / len(unique_labels)
