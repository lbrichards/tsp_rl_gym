import os
import json
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

def create_log_dir(base_dir: Path, experiment_name: str, config: Dict[str, Any]) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{experiment_name}"
    log_dir = base_dir / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = log_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Created log directory: {log_dir}")
    return log_dir


def create_experiment_directory(arm: str, instance: str, seed: int, base_dir: str = "outputs") -> str:
    """
    Creates a standardized, timestamped directory for an experiment run.

    Format: {base_dir}/{arm}/{YYYYMMDD-HHMMSS}_{instance}_seed{seed}/

    Args:
        arm (str): The name of the experimental arm (e.g., "GA", "PPO").
        instance (str): The name of the problem instance (e.g., "pr76").
        seed (int): The random seed used for the run.
        base_dir (str): The root directory for all outputs.

    Returns:
        str: The path to the created directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Sanitize instance name by removing file extension
    instance_name = os.path.splitext(instance)[0]
    dir_name = f"{timestamp}_{instance_name}_seed{seed}"
    
    # Create the full path
    full_path = os.path.join(base_dir, arm, dir_name)
    
    # Create the directory
    os.makedirs(full_path, exist_ok=True)
    
    print(f"Created experiment directory: {full_path}")
    return full_path