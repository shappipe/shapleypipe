for mcts in 25 50 75 100 150 200 300 500; do
  for dataset in \
      abalone ada_prior avila connect-4 eeg google house_prices \
      jungle_chess micro mozilla4 obesity page-blocks pbcseq pol \
      run_or_walk shuttle uscensus wall-robot-nav cmc diabetes \
      jm1 kc1 kc2 pc4 phoneme qsar-biodeg vehicle
  do
    python shapley_exp/experiment/grid_sv_p_hie_mcts.py \
      --out shapley_exp/sensitivity/saves/n_samples_sensitivity.tsv \
      --mcts "$mcts" --len 6 --seed 42 --use "$dataset" --no_dump
  done
done
