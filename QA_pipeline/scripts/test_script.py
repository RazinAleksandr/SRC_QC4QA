import click
import wandb
import yaml
import torch

# from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification
from yaml import CLoader

import sys
import os
sys.path.append("/home/st-aleksandr-razin/workspace/SRC_QC4QA/")

from QA_pipeline.utils import load_model, set_random_seed
from QA_pipeline.data import make_inference_dataset
from QA_pipeline.models import eval_model

os.environ["WANDB_CONFIG_DIR"] = "/home/st-aleksandr-razin/tmp"
wandb.login(key="7b05886251cc6b183079b0926f463890604799a7")

@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=CLoader)

    model_run_name = f"test-{config['eval']['generate_config']['max_new_tokens']}-{config['eval']['generate_config']['temperature']}" + f"{config['eval']['model']['peft_model_id'].split('/')[-1]}"[5:] + '_LoRa'

    config['eval']['data']['dataset_name'] += config['run_config']['domain']
    
    config['log_config']['dir'] = config['log_config']['dir'] + config['run_config']['domain']
    config['log_config']['file_name'] = model_run_name + '.csv'
    
    config['wandb_config']['name'] = model_run_name
    config['wandb_config']['tags'] += [config['run_config']['domain'][:-6]]
    config['wandb_config']['tags'] += [config['run_config']['adapter']]

    # print(config['eval']['data']['dataset_name'])
    # print(config['log_config']['dir'])
    # print(config['log_config']['filename'])
    # print(config['wandb_config']['name'])
    # print(config['wandb_config']['tags'])

    model, tokenizer = load_model(config["eval"]["model"])
    model.eval()
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    run = wandb.init(
        **config["wandb_config"],
        config=config["eval"],
    )

    set_random_seed(config["eval"]["seed"])

    # accelerator = Accelerator()

    columns_to_save = config["eval"]["data"]['columns_to_save']

    test_dataset = make_inference_dataset(tokenizer=tokenizer, **config["eval"]["data"])
    dataset_qa = test_dataset.remove_columns(["input_ids", "attention_mask"])
    
    test_dataset = test_dataset.remove_columns(columns_to_save)

    dataloader_ids = DataLoader(
        test_dataset,
        batch_size=config["eval"]["batch_size"],
        collate_fn=DataCollatorForTokenClassification(
            tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ),
    )

    dataloader_qa = DataLoader(dataset_qa, batch_size=config["eval"]["batch_size"])

    # model, dataloader = accelerator.prepare(model, dataloader)

    eval_model(
        run,
        model,
        dataloader_ids,
        dataloader_qa,
        tokenizer,
        config["eval"]["generate_config"],
        config["log_config"],
        config["eval"]["compute_metrics"],
        columns_to_save
    )

    run.finish()


if __name__ == "__main__":
    main()
