export TASK_NAME=mrpc

python3 ml/train_bigbird.py \
  --model_name_or_path google/bigbird-roberta-base \
  --task_name ${TASK_NAME} \
  --max_seq_length 4096 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --eval_steps 100 \
  --output_dir ./checkpoints/$TASK_NAME/
