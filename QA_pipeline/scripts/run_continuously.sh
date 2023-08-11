#!/bin/bash

# Define paths
CONFIG_PATH=/home/st-aleksandr-razin/workspace/SRC_QC4QA/QA_pipeline/config/train_config.yaml
TRAINING_SCRIPT_PATH=/home/st-aleksandr-razin/workspace/SRC_QC4QA/QA_pipeline/scripts/train_script.py

# Define a list of domains
DOMAIN_CLASSES=("Networking_and_APIs" "System_Administration_and_DevOps" "Web_Development")

# Define a list of other parameters if needed
#LEARNING_RATES=("0.0001" "0.0002" "0.0003")

# Loop over each domain
for domain_class in ${DOMAIN_CLASSES[@]}
do
    accelerate launch train_script.py --config_file $CONFIG_PATH --domain $domain_class

    # done  # Uncomment this if you add the inner loop
done
