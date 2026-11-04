# ShapleyPipe

- Supplementary material is provided in the [PDF](./ICDE_2027_round_2_ShapleyPipe_supply.pdf)
- Datasets are available at [datasets](./datasets)

## Project Structure

```
├── baselines
│   ├── greedy
│   ├── random_based
│   └── saga
├── datasets
│   ├── cc18
│   └── diffprep_dataset
├── extensive_exp
│   ├── cate_validation
│   ├── correlation
│   ├── hsp_or_loo
│   ├── pos_awareness
│   ├── resolve_shapley_loo_value_distance
│   └── sp_interaction
├── kill_shapley.sh
├── mcsp
│   ├── perm_shapley_sv_mcts.py
│   ├── saves
│   └── scripts.sh
├── README.md
├── requirements.txt
├── saves
├── shapley_exp       # main experiments
│   ├── ablation
│   ├── experiment
│   ├── figure
│   └── sensitivity
└── utils
    ├── archive
    ├── datadownloader
    ├── dataloader
    ├── data.py
    ├── datatypes.py
    ├── hierarchy
    ├── operators
    ├── pretrain_mab
    └── seed.py
```

## Datasets

Diffprep dataset is already located at `datasets/diffprep_dataset`.

Execute the following command lines to download CC18 and large datasets. They will be located at `datasets/cc18`.

```sh
python utils/datadownloader/download_cc18.py
python utils/datadownloader/download_oml.py
```

## Baselines

#### LOO

```sh
python baselines/greedy/loo.py
```

#### Random Search

```sh
python baselines/random_based/run_diffprep.py
python baselines/random_based/run_cc18.py
```

## Extensive Experiments

### Main Experiment

```sh
sh shapley_exp/experiment/script.sh
```

### Positional Sensitivity Analysis

Get all the positional sensitivity $\rho$.

```bash
python extensive_exp/hsp_or_loo/prior_check.py
```

Draw figure with `shapley_exp/figure/main.ipynb` Module `Positional Sensitivity v.s. Gain over Baseline`.

### Category Coherence Analysis

Get all the shapley values of all datasets, each operator and each position.

```sh
sh extensive_exp/cate_validation/get_each_op_sv.sh
```

Calculate correlation with

$$
\begin{aligned}\rho _{ j } \left( \phi _{ i _{ 1 } , j } ^{ \left[ 27 \right] } , \phi _{ i _{ 2 } , j } ^{ \left[ 27 \right] } \right) & = \frac{ \text{cov} \left( \phi ^{ \left[ 27 \right] } _{ i _{ 1 } , j } , \phi ^{ \left[ 27 \right] } _{ i _{ 2 } , j } \right) }{ \sigma _{ i _{ 1 } , j } \cdot \sigma _{ i _{ 2 } , j } } \\ & = \frac{ E \left[ \phi _{ i _{ 1 } , j } ^{\prime\left[ 27 \right] } \cdot \phi _{ i _{ 2 } , j } ^{\prime\left[ 27 \right] } \right] }{ \sigma _{ i _{ 1 } , j } \cdot \sigma _{ i _{ 2 } , j } }\end{aligned}
$$

Draw figure with `shapley_exp/figure/cate_corr_heatmap.py`

### MC-SP

```sh
sh mcsp/scripts.sh
```

### Extended Operators

```sh
sh shapley_exp/sensitivity/run_extended_ops.sh
```
