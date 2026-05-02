#!/usr/bin/env python3
import os
import yaml
import json
import random
import subprocess
import sys
import pandas as pd
from datetime import datetime

# --- Configuration ---
BASE_CONFIG = "configs/config.yml"
HISTORY_FILE = "autoresearch_nnunet_history.json"
RESULTS_FILE = "results_nnunet.tsv"

# Thompson Sampling families (compatible with config.yml params)
tweak_templates = [
    {"family": "lr", "attr": "initial_lr", "vals": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]},
    {"family": "wd", "attr": "weight_decay", "vals": [3e-5, 1e-4, 1e-3, 0.0]},
    {"family": "epochs", "attr": "num_epochs", "vals": [150, 300, 500, 1000]},
]

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {t["family"]: 1.0 for t in tweak_templates}

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def run_step(script_name, args=None):
    cmd = ["python3", script_name] + (args or [])
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    history = load_history()
    
    # Success-Biased Decay
    for f in history:
        history[f] = max(1.0, history[f] * 0.95)
    
    # Choose next tweak
    families = [t["family"] for t in tweak_templates]
    weights = [history[f] for f in families]
    template = random.choices(tweak_templates, weights=weights, k=1)[0]
    val = random.choice(template["vals"])
    family = template["family"]
    attr = template["attr"]
    
    print(f"--- Cycle Start: Tweaking {family} ({attr} = {val}) ---")
    
    # Load and update config
    with open(BASE_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    
    # We only optimize the FIRST configuration for now in the loop
    variant_name = f"evolved_{family}_{val}".replace(".", "p")
    config['configurations'] = [{
        'name': variant_name,
        'params': {
            'initial_lr': val if attr == 'initial_lr' else config['configurations'][0]['params'].get('initial_lr', 1e-2),
            'weight_decay': val if attr == 'weight_decay' else config['configurations'][0]['params'].get('weight_decay', 3e-5),
            'num_epochs': val if attr == 'num_epochs' else config['configurations'][0]['params'].get('num_epochs', 1000),
        }
    }]
    
    temp_config = f"configs/config_tmp.yml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
        
    # Execute Pipeline
    try:
        run_step("generate_trainers.py", ["--config", temp_config])
        run_step("run_training.py", ["--config", temp_config, "--variants", variant_name])
        # run_step("run_inference.py", ["--config", temp_config, "--variants", variant_name, ...])
        # run_step("evaluate_models.py", ["--config", temp_config, "--variants", variant_name, ...])
        
        # --- Logic for 'Success' goes here (compare Dice with baseline) ---
        # For now, we simulate a 'keep' to demonstrate the loop
        is_improvement = True # Real check: if current_dice > best_dice
        
        if is_improvement:
            history[family] += 1.0
            print(f"SUCCESS: Family {family} improved the model.")
            
        save_history(history)
        
    except Exception as e:
        print(f"Cycle crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
