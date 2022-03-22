from pathlib import Path
from typing import Sequence, Iterable, Union

import numpy as np
import torch


def accuracy_score_scalers(y_true: Sequence, y_pred: Sequence) -> float:
    """Score predictions against reference given scalar class labels,
    i.e., labels[i] ~ {0, 1}.
    """
    correct = np.sum(y_true == y_pred)
    score = correct / len(y_true)

    return score


def save_checkpoint(model_state: dict,
                    optim_state: dict,
                    file_name: Union[str, Path],
                    **params) -> None:
    """Checkpoint model params during training."""
    checkpoint = {"model_state_dict": model_state,
                  "optim_state_dict": optim_state}
    for key, val in params.items():
        checkpoint[key] = val
    torch.save(checkpoint, file_name)


def load_checkpoint(file_name: Union[str, Path]) -> dict:
    """Retrieve saved model state dict."""
    return torch.load(file_name)


def save_results(results: Sequence[int],
                 file_name: Union[str, Path] = "upload_predictions.txt") -> None:
    """Write final predictions to submission file."""
    with open(file_name, mode='w', encoding="utf-8") as f:
        for x in results:
            out = str(int(x))
            f.write(out + '\n')


if __name__ == "__main__":
    print(issubclass(Sequence, Iterable))
