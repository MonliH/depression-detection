python3 ml/eval.py \
  --model_name_or_path checkpoints/bigbird-roberta-base-filtered-more-data-2/best \
  --max_seq_length 4096 \
  --output_dir /tmp/ml \
  --load_dataset_from_disk /mnt/disks/persist/user_comments_text_filtered_2/test
