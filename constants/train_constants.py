"""
# Author: ruben 
# Date: 22/9/22
# Project: CardioEvents
# File: train_constants.py

Description: Constants regarding train process management
"""

# Architecture parameters
# =======================
from constants.path_constants import CAC_DATASET_FOLDER

DATASETS = {'CAC': {'batch_size': 8, 'class_values': {'CACSmenos400': 0, 'CACSmas400': 1}, 'path': CAC_DATASET_FOLDER,
                    'selector': [1, 1],
                    }}

CUSTOM_NORMALIZED = True  # whether image is normalized retina-based (true) or Imagenet-based (false)
MODEL_SEED = 3  # Fix seed to generate always deterministic results (same random numbers)
REQUIRES_GRAD = True  # Allow backprop in pretrained weights
WEIGHT_INIT = 'Seeded'  # Weight init . Supported --> ['KaimingUniform', 'KaimingNormal', 'XavierUniform', 'XavierNormal']

# Train hyperparameters
# =======================

EPOCHS = 150
LEARNING_RATE = 0.0001
LR_SCHEDULER = False
WEIGHT_DECAY = 4e-2
CRITERION = 'CrossEntropyLoss'
OPTIMIZER = 'SDG'

# Execution parameters
# =======================

SAVE_MODEL = False  # True if model has to be saved
SAVE_LOSS_PLOT = True  # True if loss plot has to be saved
SAVE_ACCURACY_PLOT = True  # True if accuracy plot has to be saved
ND = 2  # Number of decimals at outputs

