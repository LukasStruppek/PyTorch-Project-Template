import torch
from torch.utils.data import random_split


def get_train_val_split(data, split_ratio, seed=0):
    validation_set_length = int(split_ratio * len(data))
    training_set_length = len(data) - validation_set_length
    torch.manual_seed(seed)
    training_set, validation_set = random_split(
        data, [training_set_length, validation_set_length])

    return training_set, validation_set


def get_subsampled_dataset(dataset,
                           dataset_size=None,
                           proportion=None,
                           seed=0):
    if dataset_size > len(dataset):
        raise ValueError(
            'Dataset size is smaller than specified subsample size')
    if dataset_size is None:
        if proportion is None:
            raise ValueError('Neither dataset_size nor proportion specified')
        else:
            dataset_size = int(proportion * len(dataset))
    torch.manual_seed(seed)
    subsample, _ = random_split(
        dataset, [dataset_size, len(dataset) - dataset_size])
    return subsample
