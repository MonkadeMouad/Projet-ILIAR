import itertools
import copy
import yaml
import os

# Path to the base YAML config
CONFIG_PATH = "/usr/users/avr/avr_11/hammou1/hammou/ILIAR/configs/tmp4qmvnwmx-config.yml"

# Directory where new YAML configs will be saved
OUTPUT_DIR = "/usr/users/avr/avr_11/hammou1/hammou/ILIAR/configs/grid_search/"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the hyperparameter search space
param_grid = {
    "optim_lr": [0.00005, 0.0001, 0.0005],  # Base learning rate for optimizer
    "optim_weight_decay": [0.001, 0.01],  # Regularization strength
    "scheduler_max_lr": [0.0001, 0.0005, 0.001],  # Peak LR for OneCycleLR
    "scheduler_pct_start": [0.2, 0.3, 0.4],  # % of training spent increasing LR
    "scheduler_anneal_strategy": ["cos", "linear"],  # Decay strategy
    "scheduler_final_div_factor": [10, 100],  # Final LR scaling
}

# Load base config
with open(CONFIG_PATH, "r") as file:
    base_config = yaml.safe_load(file)

# Generate all possible combinations of hyperparameters
grid_combinations = list(itertools.product(*param_grid.values()))

yaml_files = []

# Loop through each combination and generate a YAML file
for i, values in enumerate(grid_combinations):
    # Copy base config
    config = copy.deepcopy(base_config)

    # Update optimizer parameters
    config["optim"]["params"]["lr"] = values[0]
    config["optim"]["params"]["weight_decay"] = values[1]

    # Update scheduler parameters
    config["scheduler"]["max_lr"] = values[2]
    config["scheduler"]["pct_start"] = values[3]
    config["scheduler"]["anneal_strategy"] = values[4]
    config["scheduler"]["final_div_factor"] = values[5]

    # Save new config file
    filename = f"config_{i+1}.yml"
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    with open(file_path, "w") as file:
        yaml.dump(config, file)

    yaml_files.append(file_path)

print(f"✅ {len(yaml_files)} configuration files generated in {OUTPUT_DIR}")
print("\n📂 List of generated YAML files:")
for file in yaml_files:
    print(f" - {file}")
