python3 ml/train.py \
  --model_name_or_path google/bigbird-roberta-base \
  --max_seq_length 4096 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --eval_steps 250 \
  --load_dataset_from_disk /mnt/disks/persist/user_comments_text_filtered \
  --report_to_wandb jonatli/depression/bigbird-roberta-base-filtered-more-data \
  --output_dir ./checkpoints/bigbird-roberta-base-filtered-more-data
