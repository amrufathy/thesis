export TOKENIZERS_PARALLELISM="false"
python src/finetune_bert.py \
--model_name_or_path bert-base-cased \
--train_file data/roc/roc_stories_lines_bert.txt \
--validation_file data/roc/roc_stories_lines_bert.txt \
--output_dir data/story-bert \
--do_train \
--fp16 \
--report_to wandb \
--line_by_line \
--overwrite_output_dir \
--save_strategy epoch \
--save_total_limit 1 \
--num_train_epochs 10 \
--max_steps -1 \
