def print_config(config):
    print('-'*100)
    print('EXPERIMENT CONFIGURATION')
    print('-'*100)
    print("Model Parameters:")
    print(f"  Name: {config['model_params']['model_name']}")
    print(f"  Num Classes: {config['model_params']['num_classes']}")
    print(f"  Input Dim: {config['model_params']['n_inputs']}")
    print(f"  Batch Norm: {config['model_params']['batch_norm']}")
    print(f"  Dropout Rate: {config['model_params']['dropout_rate']}")
    print(f"  Activation: {config['model_params']['activation']}")
    
    print("\nData Parameters:")
    print(f"  Data Path: {config['data_params']['data_path']}")
    print(f"  Tokenizer Name: {config['data_params']['tokenizer_name']}")
    print(f"  Max Length: {config['data_params']['max_len']}")
    print(f"  Train Batch Size: {config['data_params']['train_batch_size']}")
    print(f"  Val Batch Size: {config['data_params']['val_batch_size']}")
    print(f"  Num Workers: {config['data_params']['num_workers']}")
    print(f"  Pin Memory: {config['data_params']['pin_memory']}")
    print(f"  Transform: {config['data_params']['transform']}")
    
    print("\nExperiment Parameters:")
    print(f"  Learning Rate: {config['exp_params']['LR']}")
    print(f"  Weight Decay: {config['exp_params']['weight_decay']}")
    print(f"  Scheduler Gamma: {config['exp_params']['scheduler_gamma']}")
    print(f"  KLD Weight: {config['exp_params']['kld_weight']}")
    print(f"  Manual Seed: {config['exp_params']['manual_seed']}")
    
    print("\nTrainer Parameters:")
    print(f"  Max Epochs: {config['trainer_params']['max_epochs']}")
    
    print("\nLogging Parameters:")
    print(f"  Log Directory: {config['logging_params']['log_dir']}")
    print(f"  Save Directory: {config['logging_params']['save_dir']}")
    print(f"  Name: {config['logging_params']['name']}")
    print('-'*100)
    print(' ')

