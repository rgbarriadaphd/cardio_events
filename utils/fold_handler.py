"""
# Author: ruben 
# Date: 27/1/22
# Project: MTLFramework
# File: fold_handler.py

Description: Class to manage the change of test and train data when iteration over the input samples
"""
import logging
import os
import shutil
from distutils.dir_util import copy_tree
from constants.path_constants import CEP, CEN

FOLD_ID = 'inner_fold_'


class FoldHandler:

    def __init__(self, fold_base, run_base, criteria='5-fold'):
        """
        FoldHandler constructor to initialize fold managemet
        :param fold_base: (str) original dataset path
        :param run_base:  (str) target dataset path
        :param criteria: Cross validation criteria (onliy 5-fold suported so far)
        """
        self._fold_base = fold_base
        self._run_base = run_base
        self._nfolds = 5 if criteria == '5-fold' else 10

        logging.info(f'Running experiment based on {criteria} cross validation criteria')

    def generate_run_set(self, test_fold_id=1):
        """
        Organize folds split in train and test folds
        :param test_fold_id: Fold identifier that will be used for testing
        """

        test_set = {CEP: [], CEN: []}
        train_set = {CEP: [], CEN: []}
        if os.path.isdir(self._run_base):
            shutil.rmtree(self._run_base)
        os.mkdir(self._run_base)
        os.mkdir(os.path.join(self._run_base, 'train'))
        os.mkdir(os.path.join(self._run_base, 'train', CEP))
        os.mkdir(os.path.join(self._run_base, 'train', CEN))
        os.mkdir(os.path.join(self._run_base, 'test'))
        os.mkdir(os.path.join(self._run_base, 'test', CEP))
        os.mkdir(os.path.join(self._run_base, 'test', CEN))

        train_folds_mas = []
        train_folds_menos = []

        for fold_id in range(0, self._nfolds):
            org_mas = os.path.join(self._fold_base, FOLD_ID + str(fold_id + 1), CEP)
            org_menos = os.path.join(self._fold_base, FOLD_ID + str(fold_id + 1), CEN)
            if (fold_id + 1) == test_fold_id:
                test_set[CEP] = [item.split('.')[0] for item in os.listdir(org_mas)]
                test_set[CEN] = [item.split('.')[0] for item in os.listdir(org_menos)]
                # Copy test fold
                dst_mas = os.path.join(self._run_base, 'test', CEP)
                dst_menos = os.path.join(self._run_base, 'test', CEN)
            else:
                train_folds_mas.append(os.listdir(org_mas))
                train_folds_menos.append(os.listdir(org_menos))
                # Rest of train folds
                dst_mas = os.path.join(self._run_base, 'train', CEP)
                dst_menos = os.path.join(self._run_base, 'train', CEN)
            copy_tree(org_mas, dst_mas)
            copy_tree(org_menos, dst_menos)

        train_set[CEP] = [item.split('.')[0] for sublist in train_folds_mas for item in sublist]
        train_set[CEN] = [item.split('.')[0] for sublist in train_folds_menos for item in sublist]

        return train_set, test_set

