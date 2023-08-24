import numpy as np
from scipy import stats


def load_oeit_data(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        eit = parse_oeit_line(line)
        if eit is not None:
            data.append(eit)

    mode_len = stats.mode([len(item) for item in data], keepdims=False)
    data = [item for item in data if len(item) == mode_len.mode]

    return np.array(data)


def parse_oeit_line(line):
    try:
        _, data = line.split(":", 1)
    except (ValueError, AttributeError):
        return None
    items = []
    for item in data.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            items.append(float(item))
        except ValueError:
            return None
    return np.array(items)
