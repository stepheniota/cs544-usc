from dataclasses import dataclass, asdict, field

import numpy as np

@dataclass(frozen=True)
class Hyperparams:
    in_features: int
    out_features: int = 2
    lr: float = 1e-4
    n_epoch: int = 100
    batch_size: int = 16
    optim: str = "SGD"
    loss: str = "BCE"

    def asdict(self):
        return asdict(self)

def accuracy_score_scalers(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    score = correct / len(y_true)

    return score

def save_results(results, file_name="upload_predictions.txt"):
    with open(file_name, mode='w', encoding="utf-8") as f:
        for x in results:
            out = str(int(x))
            f.write(out + '\n')


if __name__ == "__main__":
    hparams = Hyperparams(in_features=50, out_features=2, optim="adam")
    print(asdict(hparams))
