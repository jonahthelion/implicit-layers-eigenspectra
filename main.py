from fire import Fire

import source


if __name__ == "__main__":
    Fire({
        'toy_model': source.explore.toy_model,
        'check_inverse': source.explore.check_inverse,

        'train_mlp': source.train.train_mlp,
        'train_deq': source.train.train_deq,
        'eval_model': source.train.eval_model,
        'eval_mlp_spectrum': source.train.eval_mlp_spectrum,
        'eval_deq_spectrum': source.train.eval_deq_spectrum,
        'plot_mlp_spectrum': source.train.plot_mlp_spectrum,
        'plot_deq_spectrum': source.train.plot_deq_spectrum,

        'linear_regression': source.toy.linear_regression,
        'check_deq': source.sanity_check.check_deq,
        'check_full_deq': source.deqmodel.check_full_deq,
    })