# batch sampling jobs (300 total)
for SEED in $(seq 1 12); do
  for LEN in 1024 2048; do
    sbatch ./scripts/gen_ppl/genppl_bd3lm.sh $LEN $SEED 16
    sbatch ./scripts/gen_ppl/genppl_bd3lm.sh $LEN $SEED 8
    sbatch ./scripts/gen_ppl/genppl_bd3lm.sh $LEN $SEED 4
    sbatch ./scripts/gen_ppl/genppl_mdlm.sh $LEN $SEED
  done
done

python ./scripts/log_sample_stats.py






# for SEED in $(seq 1 12); do
#   for LEN in 1024 2048; do
#     sbatch ./scripts/gen_ppl/genppl_bd3lm.sh $LEN $SEED 16
#     sbatch ./scripts/gen_ppl/genppl_bd3lm.sh $LEN $SEED 8
#     sbatch ./scripts/gen_ppl/genppl_bd3lm.sh $LEN $SEED 4
#     sbatch ./scripts/gen_ppl/genppl_mdlm.sh $LEN $SEED
#     sbatch ./scripts/gen_ppl/genppl_ar.sh $LEN $SEED    
#   done
#   sbatch ./scripts/gen_ppl/genppl_sedd.sh $SEED
# done

# python ./scripts/log_sample_stats.py
