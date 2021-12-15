from collections import defaultdict

import numpy as np
import torch

import dataset
from combine_sampler import CombineSampler


def create_loaders(dataset_name, data_root, num_classes, is_extracted, num_workers, num_classes_iter,
                   num_elements_class, size_batch):
    Dataset = dataset.load(
        name=dataset_name,
        root=data_root,
        transform=dataset.utils.make_transform(),
        labels=list(range(0, num_classes)),
        mode='train',
        is_extracted=is_extracted)
    #
    # Dataset = dataset.Birds(
    #     root=data_root,
    #     labels=list(range(0, num_classes)),
    #     is_extracted=is_extracted,
    #     transform=dataset.utils.make_transform())

    ddict = defaultdict(list)
    for idx, label in enumerate(Dataset.ys):
        ddict[label].append(idx)

    list_of_indices_for_each_class = []
    for key in ddict:
        list_of_indices_for_each_class.append(ddict[key])

    dl_tr = torch.utils.data.DataLoader(
        Dataset,
        batch_size=size_batch,
        shuffle=False,
        sampler=CombineSampler(list_of_indices_for_each_class, num_classes_iter, num_elements_class),
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    if data_root == 'Stanford':
        class_end = 2 * num_classes - 2
    else:
        class_end = 2 * num_classes

    dl_ev = torch.utils.data.DataLoader(
        dataset.load(
            name=dataset_name,
            root=data_root,
            transform=dataset.utils.make_transform(is_train=False),
            labels=list(range(num_classes, class_end)),
            mode='eval',
            is_extracted=is_extracted),
        batch_size=50,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    dl_finetune = torch.utils.data.DataLoader(
        dataset.Birds(
            root=data_root,
            labels=list(range(num_classes)),
            is_extracted=is_extracted,
            transform=dataset.utils.make_transform(is_train=False)
        ),
        batch_size=size_batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    dl_train_evaluate = torch.utils.data.DataLoader(
        dataset.Birds(
            root=data_root,
            labels=list(range(num_classes)),
            is_extracted=is_extracted,
            transform=dataset.utils.make_transform(is_train=False)
        ),
        batch_size=150,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    return dl_tr, dl_ev, dl_finetune, dl_train_evaluate


def create_loaders_finetune(data_root, num_classes, is_extracted, num_workers, size_batch):
    if data_root == 'Stanford':
        class_end = 2 * num_classes - 2
    else:
        class_end = 2 * num_classes

    dl_ev = torch.utils.data.DataLoader(
        dataset.Birds(
            root=data_root,
            labels=list(range(num_classes, class_end)),
            is_extracted=is_extracted,
            transform=dataset.utils.make_transform(is_train=False)
        ),
        batch_size=150,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    dl_finetune = torch.utils.data.DataLoader(
        dataset.Birds(
            root=data_root,
            labels=list(range(num_classes)),
            is_extracted=is_extracted,
            transform=dataset.utils.make_transform(is_train=False)
        ),
        batch_size=size_batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return dl_ev, dl_finetune


def get_labeled_and_unlabeled_points(labels, num_points_per_class, num_classes=100):
    labs, L, U = [], [], []
    labs_buffer = np.zeros(num_classes)
    num_points = labels.shape[0]
    for i in range(num_points):
        if labs_buffer[labels[i]] == num_points_per_class:
            U.append(i)
        else:
            L.append(i)
            labs.append(labels[i])
            labs_buffer[labels[i]] += 1
    return labs, L, U


def debug_info(gtg, model):
    for name, param in gtg.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                print(name, torch.mean(param.grad.data))
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                print(name, torch.mean(param.grad.data))
    print("\n\n\n")
