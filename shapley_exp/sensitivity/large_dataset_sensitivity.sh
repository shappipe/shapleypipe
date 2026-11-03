for dataset in higgs covertype airlines; do
  python shapley_exp/experiment/grid_sv_p_hie_mcts.py \
    --out shapley_exp/sensitivity/saves/large_dataset.tsv \
    --len 6 --seed 42 --use "$dataset" --limit 20000
done
