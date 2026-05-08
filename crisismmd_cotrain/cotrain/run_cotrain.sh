#!/usr/bin/env bash

for i in 7 8; do
  echo "Running with labeled_sample_idx=$i"
  python3 main_bertweet.py \
    --exp_name bert-tweet-huma-ex-crctd \
    --dataset humanitarian \
    --labeled_sample_idx $i \
    --hf_model_id_short llama-3-8b \
    --seed 1234 \
    --plm_id bert-tweet \
    --setup_local_logging
done

for i in 7 8 9; do
  echo "Running with labeled_sample_idx=$i"
  python3 main_bertweet.py \
    --exp_name bert-tweet-info-ex-crctd \
    --dataset informative \
    --labeled_sample_idx $i \
    --hf_model_id_short llama-3-8b \
    --seed 1234 \
    --plm_id bert-tweet \
    --setup_local_logging
done

for i in 7 8; do
  echo "Running with labeled_sample_idx=$i"
  python3 main_bharani.py \
    --exp_name clip-text-huma-ex-crctd \
    --dataset humanitarian \
    --labeled_sample_idx $i \
    --hf_model_id_short llama-3-8b \
    --seed 1234 \
    --plm_id clip \
    --setup_local_logging
done

# for i in 0 1 2 3 4 5 6 7 8 9; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 main_bharani.py \
#     --exp_name clip-text-info-ex-crctd \
#     --dataset informative \
#     --labeled_sample_idx $i \
#     --hf_model_id_short llama-3-8b \
#     --seed 1234 \
#     --plm_id clip \
#     --setup_local_logging
# done
