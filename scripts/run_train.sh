WANDB_PROJECT=SO_LLAMA
WANDB_WATCH=all
WANDB_SILENT=false
WANDB_LOG_MODEL=checkpoint

TOKENIZERS_PARALLELISM=false
accelerate launch train_script.py --config_file /home/st-gorbatovski/sollama/src/sft/config/train_config.yaml