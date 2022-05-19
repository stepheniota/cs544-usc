import csv
from dataclasses import dataclass
from pathlib import Path
from typing import (Union,)


@dataclass
class PNLIData:
    precondition: str
    statement: str
    label: int


def read_data(
    *files: Union[Path, str],
) -> list[tuple[str, str, int]]:
    for file in files:
        with open(file, mode='r') as f:
            reader = csv.reader(f)
            data = list(tuple(line) for line in reader)

    return data
