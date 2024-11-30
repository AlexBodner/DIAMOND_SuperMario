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
	keys: List[int]  # keys es una lista de las teclas que se presionan, se representan como enteros
	steering_value: float  # steering value es el valor de la dirección en la que se mueve el jugador (izquierda o derecha), va entre -1 y 1

	@property
	def key_names(self) -> List[str]:
		return [pygame.key.name(key) for key in self.keys]
			
	


def print_csgo_action(action: CSGOAction) -> Tuple[str]:
	action_names = [CSGO_KEYMAP[k] for k in action.keys] if len(action.keys) > 0 else []
	keys = " + ".join(action_names)
	return f"{keys} [Steering {action.steering_value}] "
	

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
	keys_set = set(csgo_action.keys)
	action_vector = np.zeros(7)
	
	if 'd' in keys_set and 'w' in keys_set:    # derecha y saltar
		action_vector[2] = 1
	elif 'd' in keys_set and 'f' in keys_set:  # derecha y correr
		action_vector[3] = 1
	elif 'd' in keys_set:                      # derecha
		action_vector[1] = 1
	elif 'a' in keys_set:                      # izquierda
		action_vector[6] = 1
	elif 'w' in keys_set:                      # saltar
		action_vector[5] = 1
	else:
		action_vector[0] = 1                   # no hacer nada

	return torch.tensor(
		action_vector,
		device=device,
		dtype=torch.float32,
	)
	

def decode_csgo_action(y_preds: torch.Tensor) -> CSGOAction:
	y_preds = y_preds.squeeze()
	steering_vector = y_preds[0:22]
	boosting = y_preds[-1]
	print(steering_vector)
	onehot_index = steering_vector.argmax()
	steering_value = index_to_decimal(onehot_index)

	keys_pressed = []
	if boosting == 1:
		keys_pressed.append("d")
	
	keys_pressed = [pygame.key.key_code(x) for x in keys_pressed]

	return CSGOAction(keys_pressed, steering_value)

