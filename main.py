from fire import Fire

import source


if __name__ == "__main__":
    Fire({
        'toy_model': source.explore.toy_model,
        'check_inverse': source.explore.check_inverse,

        'train_mlp': source.train.train_mlp,
        'eval_model': source.train.eval_model,
        'eval_mlp_spectrum': source.train.eval_mlp_spectrum,
        'plot_mlp_spectrum': source.train.plot_mlp_spectrum,

        'linear_regression': source.toy.linear_regression,
        'check_deq': source.sanity_check.check_deq,
    })