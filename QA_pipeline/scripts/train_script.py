import click
import yaml
from yaml import CLoader
import wandb
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    get_peft_model_state_dict,
)
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
)

from transformers import (
    # GenerationConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

import sys
import os

sys.path.append("/home/st-aleksandr-razin/workspace/SRC_QC4QA/")
from QA_pipeline.data import make_train_dataset  # , make_inference_dataset
from QA_pipeline.utils import load_model, set_random_seed, SavePeftModelCallback

os.environ["WANDB_PROJECT"] = "SRC_QC4QA"
os.environ["WANDB_CONFIG_DIR"] = "/home/st-aleksandr-razin/tmp"
os.environ["WANDB_WATCH"] = "none"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
wandb.login(key="7b05886251cc6b183079b0926f463890604799a7")

@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
#@click.option("--domain", default="", help="Path to config YAML file")
def main(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=CLoader)

    #config['model_run_name']['domain_class'] = domain
    model_run_name = f"train-{config['model_run_name']['main_model']}-{config['model_run_name']['adapter_type']}-{config['model_run_name']['domain_class']}-bs_16-lr_{config['training_arguments']['learning_rate']}-m_l_{config['data']['dataset']['max_length']}-m_p_l_{config['data']['dataset']['max_prompt_length']}-w_decay_{config['training_arguments']['weight_decay']}"
    print(config['model_run_name']['domain_class'])
    config['data']['dataset']['dataset_name'] += config['model_run_name']['domain_class'] + '_class'
    config['training_arguments']['output_dir'] = config['training_arguments']['output_dir'] + model_run_name
    config['training_arguments']['run_name'] = model_run_name

    set_random_seed(config["training_arguments"]["seed"])

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    ddp = world_size != 1
    print(ddp)
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        config["model"]["device_map"] = device_map

    model, tokenizer = load_model(config["model"])
    if config["model"]["load_in_8bit"]:
        model = prepare_model_for_kbit_training(model)
    #print(model)
    
    # if not ddp and torch.cuda.device_count() > 1:
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    # prepare Adapter model for training
    if not config["model"].get("peft_model_id"):
        if config['model_run_name']['adapter_type'] == 'Lora':
            lora_config = LoraConfig(task_type="CAUSAL_LM", **config["lora_config"])
            model = get_peft_model(model, lora_config)
        else:
            peft_config = PromptEncoderConfig(peft_type="P_TUNING", task_type="CAUSAL_LM", **config["ptune_config"])
            model = get_peft_model(model, peft_config)
    
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # make datasets
    train_dataset = make_train_dataset(
        tokenizer=tokenizer, split=config["data"]["train_split"], **config["data"]["dataset"]
    )
    eval_dataset = make_train_dataset(
        tokenizer=tokenizer, split=config["data"]["val_split"], **config["data"]["dataset"]
    )

    # make generatoin config
    # generation_config = GenerationConfig(**config["generation_config"])

    # training_args = Seq2SeqTrainingArguments(generation_config=generation_config, **config["training_argiments"])
    training_args = TrainingArguments(
        ddp_find_unused_parameters=False if ddp else None,
        **config["training_arguments"],
    )

    callbacks = [SavePeftModelCallback] if config["lora_config"] else []

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        data_collator=DataCollatorForTokenClassification(
            tokenizer,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=8,
        ),
    )
    if config["training_arguments"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    model.config.use_cache = not config["training_arguments"]["gradient_checkpointing"]

    if config["model"]["load_in_8bit"]:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(config["training_arguments"]["resume_from_checkpoint"])
    tokenizer.save_pretrained(config["training_arguments"]["output_dir"])
    model.save_pretrained(config["training_arguments"]["output_dir"])


if __name__ == "__main__":
    main()
