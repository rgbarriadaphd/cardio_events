"""
# Author: ruben 
# Date: 22/9/22
# Project: CardioEvents
# File: cnn.py

Description: Functions to deal with the cnn operations
"""
import logging
import math

import torch
from torch import nn
from torchvision import models
from torch import optim
from tqdm import tqdm
from constants.train_constants import *
from utils.metrics import PerformanceMetrics


class DLModel:
    """
    Class to manage the architecture initialization
    """

    def __init__(self, device, path=None):
        """
        Architecture class constructor
        :param device: (torch.device) Running device
        """
        self._device = device
        self._model = None
        n_classes = sum([len(DATASETS[dt]['class_values']) for dt in DATASETS])

        if MODEL_SEED > 0:
            torch.manual_seed(MODEL_SEED)

        self._model = models.vgg16(pretrained=True)

        num_features = self._model.classifier[6].in_features
        features = list(self._model.classifier.children())[:-1]  # Remove last layer
        linear = nn.Linear(num_features, n_classes)
        features.extend([linear])
        self._model.classifier = nn.Sequential(*features)

        for param in self._model.parameters():
            param.requires_grad = REQUIRES_GRAD

    def get(self):
        """
        Return model
        """
        return self._model


def train_model(model, device, train_loaders):
    """
    Trains the model with input parametrization
    :param model: (torchvision.models) Pytorch model
    :param device: (torch.cuda.device) Computing device
    :param train_loaders: (list torchvision.datasets) List of  train dataloader containing dataset images
    :return: train model, losses array, accuracies of test and train datasets
    """
    n_train = sum([len(train_loaders[i].dataset) for i in range(len(train_loaders))])

    logging.info(f'''Starting training:
            Epochs:          {EPOCHS}
            Batch size:      {[(dt, DATASETS[dt]['batch_size']) for dt, values in DATASETS.items()]}
            Learning rate:   {LEARNING_RATE}
            Training sizes:   {[(train_loaders[i].dataset.dataset_name, len(train_loaders[i].dataset))
                                for i in range(len(train_loaders))]}
            Device:          {device.type}
            Criterion:       {CRITERION}
            Optimizer:       {OPTIMIZER}
        ''')

    losses = []
    test_accuracy_list = []
    train_accuracy_list = []

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    inf_condition = False
    for epoch in range(EPOCHS):
        model.train(True)
        running_loss = 0.0
        print(f'------------- EPOCH: {epoch + 1} -------------')

        dataset_iterator = train_loaders[0]

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:
            for i, batch in enumerate(dataset_iterator):
                if inf_condition:
                    logging.info(f'inner INF condition at epoch {epoch + 1}')
                    break
                sample, ground, index, dt_name = batch
                sample = sample.to(device=device, dtype=torch.float32)
                ground = ground.to(device=device, dtype=torch.long)

                current_batch_size = sample.size(0)
                optimizer.zero_grad()
                prediction = model(sample)

                loss = criterion(prediction, ground)

                loss.backward()
                optimizer.step()
                if math.isnan(loss.item()):
                    raise ValueError

                running_loss += loss.item() * current_batch_size

                pbar.set_postfix(**{'loss (batch) ': loss.item()})
                pbar.update(current_batch_size)

                if math.isinf(loss.item()):
                    inf_condition = True
                    break

        epoch_loss = running_loss / n_train
        print(f'EPOCH Loss : {epoch_loss}')
        losses.append(epoch_loss)

    return model, (losses, train_accuracy_list, test_accuracy_list)


def evaluate_model(model, device, test_loaders):
    """
    Test the model with input parametrization
    :param model: (torch) Pytorch model
    :param device: (torch.cuda.device) Computing device
    :param test_loaders: (List torchvision.datasets) List of  train dataloader containing dataset images
    :return: (dict) model accuracy
    """
    n_test = sum([len(test_loaders[i].dataset) for i in range(len(test_loaders))])
    logging.info(f'''Starting MTL testing:
                Test sizes:   {[(test_loaders[i].dataset.dataset_name, len(test_loaders[i].dataset))
                                for i in range(len(test_loaders))]}
                Device:          {device.type}
            ''')

    correct = 0
    total = 0
    ground_array = []
    prediction_array = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loaders[0]):
            sample, ground, index, dt_name = batch
            sample = sample.to(device=device, dtype=torch.float32)
            ground = ground.to(device=device, dtype=torch.long)

            outputs = model(sample)
            _, predicted = torch.max(outputs.data, 1)

            ground_array.append(ground.item())
            prediction_array.append(predicted.item())

            total += ground.size(0)
            correct += (predicted == ground).sum().item()

    pm = PerformanceMetrics(ground=ground_array,
                            prediction=prediction_array,
                            percent=True,
                            formatted=True)
    confusion_matrix = pm.confusion_matrix()

    performance = {
        f'accuracy': pm.accuracy(),
        f'precision': pm.precision(),
        f'recall': pm.recall(),
        f'f1': pm.f1(),
        f'tn': confusion_matrix[0],
        f'fp': confusion_matrix[1],
        f'fn': confusion_matrix[2],
        f'tp': confusion_matrix[3]
    }
    return performance
