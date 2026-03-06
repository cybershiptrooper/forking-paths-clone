import yaml
import json
import os
import copy
from typing import Dict, Any


def load_config(config_path: str) -> Dict[Any, Any]:
    """Load configuration from a YAML or JSON file, with base_config support"""
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check for base configuration
        if "base_config" in config:
            base_path = config.pop("base_config")
            # Load base config
            base_config = load_config(base_path)

            # Merge configurations (base first, then override with current)
            merged_config = copy.deepcopy(base_config)

            def deep_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        deep_update(d[k], v)
                    else:
                        d[k] = v

            deep_update(merged_config, config)
            return merged_config

        return config
    elif config_path.endswith(".json"):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Config file must be either .yaml, .yml, or .json")
