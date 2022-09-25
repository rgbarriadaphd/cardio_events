"""
# Author: ruben 
# Date: 22/9/22
# Project: CardioEvents
# File: path_constants.py

Description: Constants regarding path management
"""
import os

# Main folder structure
# =======================
OUTPUT_FOLDER = 'output/'
assert os.path.exists(OUTPUT_FOLDER)

INPUT_FOLDER = 'input/'
assert os.path.exists(OUTPUT_FOLDER)

LOGS_FOLDER = os.path.join(OUTPUT_FOLDER, 'log')
assert os.path.exists(LOGS_FOLDER)
TRAIN_FOLDER = os.path.join(OUTPUT_FOLDER, 'train')
assert os.path.exists(TRAIN_FOLDER)

CE_DATASET_FOLDER = os.path.join(INPUT_FOLDER, 'CE')
assert os.path.exists(CE_DATASET_FOLDER)

DYNAMIC_RUN_FOLDER = os.path.join(CE_DATASET_FOLDER, 'dynamic_run')
assert os.path.exists(DYNAMIC_RUN_FOLDER)


CAC_NEGATIVE = 'CACSmenos400'
CAC_POSITIVE = 'CACSmas400'
TRAIN = 'train'
TEST = 'test'

# Templates
# =======================
SUMMARY_TEMPLATE = 'templates/summary.tpl'