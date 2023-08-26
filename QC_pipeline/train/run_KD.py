import argparse
import warnings
import yaml
from typing import Dict, Any

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl

import nltk
nltk.download('stopwords')

from QC_pipeline.data.tl_dataset import E5Dataset
from QC_pipeline.models.transformers.E5_base import TextClassifier
from QC_pipeline.models.transformers.E5_small import SmallTextClassifier
from QC_pipeline.train.E5_KD_exp import E5Xperiment

warnings.filterwarnings('ignore')


def main(config_path: str) -> None:
    # Read the configuration file
    with open(config_path, 'r') as config_file:
        config: Dict[str, Any] = yaml.safe_load(config_file)

    #print_config(config)
    pl.seed_everything(config["seed"])
    # Initialize your experiment/model
    experiment = E5Xperiment(
        TextClassifier(**config['teacher_model_params']),
        SmallTextClassifier(**config['student_model_params']),
        config['exp_params']
        )
    
    # Initialize your dataset
    data = E5Dataset(
        **config["data_params"], 
        )
    data.setup()
    
    # Initialize your logger
    model_run_name = f"train-kdistil-multilabel-code-lr_{config['exp_params']['LR']}-dp_{config['teacher_model_params']['dropout_rate']}-bs_{config['data_params']['train_batch_size']}-sh_exp_{config['exp_params']['scheduler_gamma']}-wd_{config['exp_params']['weight_decay']}-temp_{config['exp_params']['temperature']}-dw_{config['exp_params']['distill_weight']}"
    wandb_logger = WandbLogger(project='SRC_QC4QA', name=model_run_name, log_model='all')

    # Initialize your callbacks
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="t_ce_val_loss",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor()

    # Initialize your trainer
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[lr_monitor, checkpoint_callback],
        **config['trainer_params']
    )

    # Train your model
    trainer.fit(experiment, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)
