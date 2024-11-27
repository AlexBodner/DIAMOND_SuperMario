"""
Credits: some parts are taken and modified from the file `config.py` from https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning/
"""

from dataclasses import dataclass 
from typing import Dict, List, Set, Tuple

import numpy as np
import pygame
import torch

from .keymap import CSGO_KEYMAP


@dataclass
class CSGOAction:
	keys: List[int]
	steering_value: float

	@property
	def key_names(self) -> List[str]:
		return [pygame.key.name(key) for key in self.keys]
			
	


def print_csgo_action(action: CSGOAction) -> Tuple[str]:
	action_names = [CSGO_KEYMAP[k] for k in action.keys] if len(action.keys) > 0 else []
	keys = " + ".join(action_names)
	return f"{keys} [Steering {action.steering_value}] | MAP: {action.map_id}"
	

def decimal_to_index(decimal_value):
    """
    Converts a decimal value to its corresponding index based on the table.

    Args:
        decimal_value (float): The decimal value to convert.

    Returns:
        int: The corresponding index, or None if the value is not in the range.
    """
    decimals = [-1.0 + 0.1 * i for i in range(21)]
    if decimal_value in decimals:
        return decimals.index(decimal_value)
    return None

def index_to_decimal(index):
    """
    Converts an index to its corresponding decimal value based on the table.

    Args:
        index (int): The index to convert.

    Returns:
        float: The corresponding decimal value, or None if the index is out of range.
    """
    if 0 <= index < 21:
        return -1.0 + 0.1 * index
    return None



def encode_csgo_action(csgo_action: CSGOAction, device: torch.device) -> torch.Tensor:

	input_vector = np.zeros(1)
	steering_vector = np.zeros(21)
	
	for key in csgo_action.key_names:
		if key == "d":
			input_vector[0] = 1
		#could iterate over more keys here
	
	steering_vector[decimal_to_index(csgo_action.steering_value)] = 1

	return torch.tensor(
		np.concatenate((
			steering_vector,
			input_vector
		)),
		device=device,
		dtype=torch.float32,
	)
	

def decode_csgo_action(y_preds: torch.Tensor) -> CSGOAction:
	y_preds = y_preds.squeeze()
	steering_vector = y_preds[0:22]
	boosting = y_preds[-1]

	onehot_index = steering_vector.index(1)
	steering_value = index_to_decimal(onehot_index)

	keys_pressed = []
	if boosting == 1:
		keys_pressed.append("d")
	
	keys_pressed = [pygame.key.key_code(x) for x in keys_pressed]

	return CSGOAction(keys_pressed, steering_value)

