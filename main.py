from fire import Fire

import source


if __name__ == "__main__":
    Fire({
        'toy_model': source.explore.toy_model,
        'check_inverse': source.explore.check_inverse,
        'train_deq': source.train.train_deq,
    })