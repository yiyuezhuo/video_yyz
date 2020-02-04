'''
wrap frozen_xyz.__dict__ logic
'''

from . import frozen_datasets, frozen_transforms, frozen_collate, frozen_dataloader
from . import frozen_models, frozen_optimizer, frozen_scheduler


def get_pipeline(root, dataset_name, transform_name, collate_name, dataloader_name):
    dataset_builder = frozen_datasets.__dict__[dataset_name]
    transform_builder = frozen_transforms.__dict__[transform_name]
    collate_builder = frozen_collate.__dict__[collate_name]
    dataloader_builder = frozen_dataloader.__dict__[dataloader_name]

    transform = transform_builder()
    dataset = dataset_builder(root, transform)
    collate = collate_builder()
    dataloader = dataloader_builder(dataset, collate)

    return dataset, transform, collate, dataloader


def get_model(model_name):
    model_builder = frozen_models.__dict__[model_name]
    model = model_builder()
    return model


def get_optimizer(optimizer_name, model_parameters):
    optimizer_builder = frozen_optimizer.__dict__[optimizer_name]
    optimizer = optimizer_builder(model_parameters)
    return optimizer


def get_scheduler(scheduler_name, data_loader_train, optimizer):
    scheduler_builder = frozen_scheduler.__dict__[scheduler_name]
    scheduler = scheduler_builder(data_loader_train, optimizer)
    return scheduler