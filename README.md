# implicit-layers-eigenspectra
Measure the eigen spectra of implicit models (e.g. DEQ, ODE)

```
git clone --recurse-submodules https://github.com/jonahthelion/implicit-layers-eigenspectra.git
```

## Test Implicit Hessian Vector Product
- `python main.py check_deq`
- `python main.py check_full_deq`
- `python main.py toy_model`

## Linear Regression
- `python main.py linear_regression`

## Deep Linear Models
1) MLP
- train: `python main.py train_mlp`
- evaluate accuracy: `python main.py eval_model ./storage/mnist/mlp0010000.pkl`
- compute eigenspectra: `python main.py eval_mlp_spectrum`
- plot eigenspectra: `python main.py plot_mlp_spectrum`
2) DEQ (parameter init)
- train: `python main.py train_deq`
- evaluate accuracy: `python main.py eval_model ./storage/mnist/deq0010000.pkl`
- compute eigenspectra: `python main.py eval_deq_spectrum --deq_init=False`
- plot eigenspectra: `python main.py plot_deq_spectrum --deq_init=False`
3) DEQ (functional init)
- train: `python main.py train_deq_init`
- evaluate accuracy: `python main.py eval_model ./storage/mnist/deqinit0010000.pkl`
- compute eigenspectra: `python main.py eval_deq_spectrum --deq_init=True`
- plot eigenspectra: `python main.py plot_deq_spectrum --deq_init=True`

## Small Non-linear Models
- In `./source/deqmodel.py`, change `h = mod(h) + input_embs` to `h = jax.nn.gelu(mod(h)) + input_embs`
- In `./source/train.py`, change `net_fn` to be an MLP with hidden size `15` and `jax.nn.gelu` non-linearity
- Re-run all commands from "Deep Linear Models" with flag `--hidden_size=15 --max_steps=50` for all DEQ commands. Note the `.pkl` files will overwrite any files already in the `./storage/mnist` folder.
