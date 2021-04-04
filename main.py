from fire import Fire

import src


if __name__ == "__main__":
    Fire({
        'toy_model': src.explore.toy_model,
        'check_inverse': src.explore.check_inverse,
    })