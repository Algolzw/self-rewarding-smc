for LEN in 1025 2050; do
  for TOTAL_T in 1000 25; do
    for SEED in $(seq 1 300); do
      sbatch run_generate_text.sh $SEED $LEN $TOTAL_T
    done
  done
done