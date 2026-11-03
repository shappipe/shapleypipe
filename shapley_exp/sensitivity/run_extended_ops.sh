export SHAPLEY_EXT_OPS=1

for seed in 42 114 256 512 768; do
	for dataset in \
		abalone ada_prior avila connect-4 eeg google house_prices \
		jungle_chess micro mozilla4 obesity page-blocks pbcseq pol \
		run_or_walk shuttle uscensus wall-robot-nav cmc diabetes \
		jm1 kc1 kc2 pc4 phoneme qsar-biodeg vehicle
	do
		python shapley_exp/ablation/grid_sv_p_hie_mcts_ext_cate.py \
			--len 6 --seed "$seed" --use "$dataset"
	done
done
