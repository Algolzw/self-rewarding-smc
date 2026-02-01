# batch sampling jobs (500 total)
MAX_LEN=131_000 # as in owt
for SEED in $(seq 1 20); do
  sbatch ./scripts/var_len/varlen_bd3lm.sh $MAX_LEN $SEED 16
  sbatch ./scripts/var_len/varlen_ar.sh $MAX_LEN $SEED
  
  # sedd does not support arbitrary-length sampling;
  # this script stops sampling when [EOS] is generated
  # and records the document length
  sbatch ./scripts/var_len/recordlen_sedd.sh $SEED
done

python scripts/log_sample_stats.py