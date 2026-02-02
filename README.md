# ShapleyPipe

Copy `example.env` to `.env` and change the values as needed.

## Project Structure

```
aipipe_reprod/
    actions/
        q_action_provider.py   # Operator Factory
        dataset_type.py        # Dataset Type
    primitives/                # Implementation of all operators
    utils/dataloader/          # Data Loader
    shapley/                   # Shapley Pipe
```

## Quick Start

Full enumeration

```
python aipipe_reprod/shapley/pipesv/grid_sv_perm_forecon_dt.p.py \
    [--set diffprep/cc18] --use <dataset_name> \
    --len 6 \
    [--n_processes 4]
    --early_stop
    --dt <lr/rf/gb/mlp>
```

With Monte Carlo

```
python aipipe_reprod/shapley/sequential_sv/sv_perm_fore_mcts.py \
    [--set diffprep/cc18] --use <dataset_name> \
    --len 6 \
    [--n_processes 4]
    --early_stop
```

Hierarchical

```
python aipipe_reprod/shapley/pipesv/grid_sv_p_forecon_mcts.py \
    [--set diffprep/cc18] --use <dataset_name> \
    --len 6 \
    [--n_processes 4]
```
