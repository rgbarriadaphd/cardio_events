"""
# Author: ruben 
# Date: 22/9/22
# Project: CardioEvents
# File: train_constants.py

Description: Constants regarding train process management
"""

# Architecture parameters
# =======================
from constants.path_constants import DYNAMIC_RUN_FOLDER

DATASETS = {'CE': {'batch_size': 8, 'class_values': {'CEN': 0, 'CEP': 1}, 'path': DYNAMIC_RUN_FOLDER,
                   'selector': [1, 1],
                   }}

CUSTOM_NORMALIZED = True  # whether image is normalized retina-based (true) or Imagenet-based (false)
MODEL_SEED = 3  # Fix seed to generate always deterministic results (same random numbers)
REQUIRES_GRAD = True  # Allow backprop in pretrained weights
WEIGHT_INIT = 'Seeded'  # Weight init . Supported --> ['KaimingUniform', 'KaimingNormal', 'XavierUniform', 'XavierNormal']

# Train hyperparameters
# =======================

EPOCHS = 1
LEARNING_RATE = 0.0001
LR_SCHEDULER = False
WEIGHT_DECAY = 4e-2
CRITERION = 'CrossEntropyLoss'
OPTIMIZER = 'SDG'

# Execution parameters
# =======================
CONTROL_TRAIN = False  # Runs model on train/test every epoch
SAVE_MODEL = False  # True if model has to be saved
SAVE_LOSS_PLOT = True  # True if loss plot has to be saved
SAVE_ACCURACY_PLOT = True  # True if accuracy plot has to be saved
ND = 2  # Number of decimals at outputs

N_INCREASED_FOLDS = 1