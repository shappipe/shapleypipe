for dataset in \
    abalone ada_prior avila connect-4 eeg google house_prices \
    jungle_chess micro mozilla4 obesity page-blocks pbcseq pol \
    run_or_walk shuttle uscensus wall-robot-nav cmc diabetes \
    jm1 kc1 kc2 pc4 phoneme qsar-biodeg vehicle
do
    python extensive_exp/cate_validation/grid_sv_dt_hypo.py \
        --len 6 --use "$dataset"
done