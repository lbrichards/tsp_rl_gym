"""
This module provides utilities for loading TSP instances from standard formats.
"""
import numpy as np
import re
from typing import Dict, Any

def load_tsplib_instance(file_path: str) -> Dict[str, Any]:
    """
    Parses a TSP instance file in the TSPLIB format.

    This function reads a .tsp file, extracts key metadata, and parses the
    node coordinates into a NumPy array.

    Args:
        file_path: The path to the .tsp file.

    Returns:
        A dictionary containing the instance's name, dimension, and a
        NumPy array of its coordinates.
    """
    instance = {}
    coords = []
    in_coord_section = False

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Use regex to find key: value pairs
            match = re.match(r'([A-Z_]+)\s*:\s*(.*)', line)
            if match:
                key, value = match.groups()
                instance[key.lower()] = value.strip()
                continue

            if line == "NODE_COORD_SECTION":
                in_coord_section = True
                continue
            
            if in_coord_section:
                if line in ("EOF", "TOUR_SECTION"):
                    break
                
                parts = line.split()
                if len(parts) >= 3:
                    # TSPLIB format is: node_id x_coord y_coord
                    coords.append([float(parts[1]), float(parts[2])])

    if "dimension" in instance:
        instance["dimension"] = int(instance["dimension"])
        
    instance["coords"] = np.array(coords, dtype=np.float32)
    return instance