"""
# Author: ruben 
# Date: 22/9/22
# Project: CardioEvents
# File: train_model.py

Description: Class to handle train stages
"""
import time
from datetime import timedelta, datetime
import csv
from string import Template

import torch
import logging

from constants.train_constants import *
from constants.path_constants import *
from dataset.cac_dataset import load_and_transform_data, get_custom_normalization
from utils.cnn import DLModel, train_model, evaluate_model
from utils.metrics import CrossValidationMeasures
from utils.fold_handler import FoldHandler


class TrainCACModel:

    def __init__(self, date_time):
        """
        Model train constructor initializes al class attributes.
        :param date_time: (str) date and time to identify execution
        """
        self._date_time = date_time
        self._model = None

        self._create_train_folder()
        self._init_device()

    def _create_train_folder(self):
        """
        Creates the folder where output data will be stored
        """
        self._train_folder = os.path.join(TRAIN_FOLDER, f'train_{self._date_time}')
        try:
            os.mkdir(self._train_folder)
        except OSError:
            logging.error("Creation of model directory %s failed" % self._train_folder)
        else:
            logging.info("Successfully created model directory %s " % self._train_folder)

    def _init_device(self):
        """
        Initialize either cuda or cpu device where train will be run
        """
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {self._device}')

    def _generate_fold_data(self, outer_fold_id):
        """
        Creates fold structure where train images will be split in 4 train folds + 1 test fold
        :param outer_fold_id: (int) Outer fold identifier
        """
        self._fold_dataset = os.path.join(ROOT_ORIGINAL_FOLDS, 'outer_fold_{}'.format(outer_fold_id))
        self._fold_handler = FoldHandler(self._fold_dataset, DYNAMIC_RUN_FOLDER)

    def _get_normalization(self):
        """
        Retrieves custom normalization if defined.
        :return: ((list) mean, (list) std) Normalized mean and std according to train dataset
        """
        # Generate custom mean and std normalization values from only train dataset
        self._normalization = get_custom_normalization() if CUSTOM_NORMALIZED else (
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _init_model(self):
        """
        Gathers model architecture
        :return:
        """
        self._model = DLModel(device=self._device).get()
        self._model.to(device=self._device)

    def _save_train_summary(self, folds_performance, global_performance):
        """
        Writes performance summnary
        :param folds_performance: (dict) Contains fold performance data
        :param global_performance: (dict) Contains global model performance data
        """
        # Global Configuration
        summary_template_values = {
            'datetime': datetime.now(),
            'model': "MTL",
            'normalized': CUSTOM_NORMALIZED,
            'save_model': SAVE_MODEL,
            'plot_loss': SAVE_LOSS_PLOT,
            'epochs': EPOCHS,
            'batch_size': [(dt, DATASETS[dt]['batch_size']) for dt, values in DATASETS.items()],
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'criterion': CRITERION,
            'optimizer': OPTIMIZER,
            'device': self._device,
            'require_grad': REQUIRES_GRAD,
            'weight_init': WEIGHT_INIT
        }

        for fold in folds_performance:
            summary_template_values.update(fold)
        summary_template_values.update(global_performance)

        # Substitute values
        tpl_file = SUMMARY_TEMPLATE.format(N_INCREASED_FOLDS)
        with open(tpl_file, 'r') as f:
            src = Template(f.read())
            report = src.substitute(summary_template_values)
            logging.info(report)

        # Write report
        with open(os.path.join(self._train_folder, 'summary.out'), 'w') as f:
            f.write(report)

    def _save_csv_data(self, train_data, outer_fold_id, inner_fold_id):
        """
        :param train_data: (tuple) lists of train data:
            train_data[0]: train loss
            train_data[1]: train accuracy overtrain dataset
            train_data[2]: train accuracy over test dataset
        :param outer_fold_id: (int) outer fold identifier
        :param inner_fold_id: (int) inner fold identifier
        """
        for i, csv_name in enumerate(['loss', 'accuracy_on_train', 'accuracy_on_test']):
            data = train_data[i]
            csv_path = os.path.join(self._train_folder, f'{csv_name}_{outer_fold_id}_{inner_fold_id}.csv')
            with open(csv_path, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerows([data])

    def run(self):
        """
        Run train stage
        """
        t0 = time.time()
        folds_performance = []
        folds_acc = []

        for outer_fold_id in range(1, N_INCREASED_FOLDS + 1):
            self._generate_fold_data(outer_fold_id)
            for inner_fold_id in range(1, 6):
                print(
                    f'***************************** Fold ID: {outer_fold_id} - {inner_fold_id} *****************************')
                # Generate fold data
                t0 = time.time()
                train_data, test_data = self._fold_handler.generate_run_set(inner_fold_id)
                print(f'Generate run set: {time.time() - t0}s')

                a = 0
                # Init model
                t0 = time.time()
                self._init_model()
                print(f'Init the model: {time.time() - t0}s')

                # Get dataset normalization mean and std
                t0 = time.time()
                self._get_normalization()
                print(f'Normalization: {time.time() - t0}s')

                # Train MTL model.
                # <--------------------------------------------------------------------->
                # Generate train data
                train_data_loaders = load_and_transform_data(stage='train',
                                                             shuffle=True,
                                                             mean=self._normalization[0],
                                                             std=self._normalization[1]
                                                             )
                t0_fold_train = time.time()
                self._model, train_data = train_model(model=self._model,
                                                      device=self._device,
                                                      train_loaders=train_data_loaders
                                                      )
                tf_fold_train = time.time() - t0_fold_train

                self._save_csv_data(train_data, outer_fold_id, inner_fold_id)

                # Test MTL model.
                # <--------------------------------------------------------------------->
                # Generate test data
                test_data_loaders = load_and_transform_data(stage='test',
                                                            mean=self._normalization[0],
                                                            std=self._normalization[1],
                                                            shuffle=False)  # shuffle does not matter for test
                t0_fold_test = time.time()
                fold_performance, fold_accuracy = evaluate_model(model=self._model,
                                                                 device=self._device,
                                                                 test_loaders=test_data_loaders,
                                                                 outer_fold_id=outer_fold_id,
                                                                 inner_fold_id=inner_fold_id)
                tf_fold_test = time.time() - t0_fold_test
                folds_acc.append(fold_accuracy)
                # Update fold data
                fold_data = {
                    f'fold_id_{outer_fold_id}_{inner_fold_id}': inner_fold_id,
                    f'n_train_{outer_fold_id}_{inner_fold_id}': [
                        (train_data_loaders[i].dataset.dataset_name, len(train_data_loaders[i].dataset)) for i in
                        range(len(train_data_loaders))],
                    f'n_test_{outer_fold_id}_{inner_fold_id}': [
                        (test_data_loaders[i].dataset.dataset_name, len(test_data_loaders[i].dataset)) for i in
                        range(len(test_data_loaders))],
                    f'mean_{outer_fold_id}_{inner_fold_id}': f'[{self._normalization[0][0]:.{ND}f}, {self._normalization[0][1]:.{ND}f}, {self._normalization[0][2]:.{ND}f}]',
                    f'std_{outer_fold_id}_{inner_fold_id}': f'[{self._normalization[1][0]:.{ND}f}, {self._normalization[1][1]:.{ND}f}, {self._normalization[1][2]:.{ND}f}]',
                    f'fold_train_time_{outer_fold_id}_{inner_fold_id}': f'{tf_fold_train:.{ND}f}',
                    f'fold_test_time_{outer_fold_id}_{inner_fold_id}': f'{tf_fold_test:.{ND}f}',
                }
                fold_performance.update(fold_data)
                folds_performance.append(fold_performance)

                l_param = ["python plot/loss_plot.py", f'"Loss evolution (fold {outer_fold_id}-{inner_fold_id})"',
                           '"epochs, loss"', '"CAC loss"',
                           f'{os.path.join(self._train_folder, "loss_" + str(outer_fold_id) + "_" + str(inner_fold_id) + ".csv")}',
                           f'{os.path.join(self._train_folder, "loss_" + str(outer_fold_id) + "_" + str(inner_fold_id) + ".png")}']

                call = ' '.join(l_param)
                os.system(call)

                if CONTROL_TRAIN:
                    l_param = ["python plot/loss_plot.py", f'"Train progress(fold {inner_fold_id})"',
                               '"epochs, accuracy"', '"train, test"',
                               f'{os.path.join(self._train_folder, "accuracy_on_train" + str(outer_fold_id) + "_" + str(inner_fold_id) + ".csv")},{os.path.join(self._train_folder, "accuracy_on_test.csv")}',
                               f'{os.path.join(self._train_folder, "accuracy" + str(outer_fold_id) + "_" + str(inner_fold_id) + ".png")}']
                    os.system(' '.join(l_param))

        # Compute global performance info
        cvm = CrossValidationMeasures(measures_list=folds_acc, percent=True, formatted=True)
        f_acc = '['
        for p, fa in enumerate(folds_acc):
            f_acc += f'{fa:.{ND}f}'
            if (p + 1) != len(folds_acc):
                f_acc += ','
        f_acc += ']'
        global_performance = {
            'execution_time': str(timedelta(seconds=time.time() - t0)),
            'folds_accuracy': f_acc,
            'cross_v_mean': cvm.mean(),
            'cross_v_stddev': cvm.stddev(),
            'cross_v_interval': cvm.interval()
        }

        self._save_train_summary(folds_performance, global_performance)
