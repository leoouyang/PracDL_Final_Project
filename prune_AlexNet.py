from train_AlexNet import *

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.utils.prune as prune
import torch.nn.functional as F

import argparse


parser = argparse.ArgumentParser(
    description='PyTorch AlexNet Prune & Retraining')
parser.add_argument('--extractor', action='store_true', default=True,
                    help='Prune feature extractor of AlexNet')
parser.add_argument('--classifier', action='store_true', default=True,
                    help='Prune feature extractor of AlexNet')
parser.add_argument('--prune_fraction', type=float, default=0.2,
                    help='Fraction of parameters to prune each iteration')
parser.add_argument('--iterations', type=int, default=3,
                    help='Number of iterations for iterative pruning')


def finetune(model, num_epochs, trainloader, testloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    test_accus = train(model, device, trainloader, testloader, criterion,
                       optimizer, num_epochs, None, scheduler=scheduler)
    train_accu = evaluate(model, trainloader, device)
    print("Finetune test accus:", test_accus)
    return test_accus[-1], train_accu


def print_mask_sum(root_module):
    mask_sum = 0
    for name,module in root_module.named_children():
        # print(name)
        for name, mask in module.named_buffers():
            # print(name, mask.sum().item())
            mask_sum += mask.sum().item()
    print("Mask sum:", mask_sum)


def print_module_weights(root_module):
    for name,module in root_module.named_children():
        print(name)
        # print(list(module.named_parameters()))
        # print(list(module.named_buffers()))
        if hasattr(module, "weight"):
            print(module.weight)


def get_parameters_to_prune(root_module, attrs2prune):
    parameters_to_prune = []
    for name, module in root_module.named_children():
        for name, param in module.named_parameters():
            if name in attrs2prune:
                parameters_to_prune.append((module, name))
    print(parameters_to_prune)
    return parameters_to_prune

if __name__ == "__main__":
    batch_size = 128
    MODEL_PATH = './alexnet_original.pth'
    CHKPT_DIR = "alexnet_chkpt"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(0)
    torch.manual_seed(0)

    trainset, testset = load_cifar10_pytorch(root='G:\ML dataset', transform=ImageNet_Transform_Func)
    # trainset, testset = load_cifar10_pytorch(transform=ImageNet_Transform_Func)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    model = models.alexnet()
    model.classifier[6] = nn.Linear(4096, 10)
    model.to(DEVICE)
    print(model)
    load_chkpt(model, MODEL_PATH, DEVICE)

    # print(evaluate(model,testloader, DEVICE)) #0.838
    # print(evaluate(model,trainloader, DEVICE)) #0.88386


    test_accu, train_accu = finetune(model, 20, trainloader, testloader, DEVICE)
    print(test_accu, train_accu)

    parameters_to_prune = get_parameters_to_prune(model.features, ("weight", "bias"))
    parameters_to_prune += get_parameters_to_prune(model.classifier, ("weight", "bias"))
    for i in range(5):
        print("=========================Iteration %i =========================="%(i+1))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.5,
        )

        print_mask_sum(model.features)
        print("Performance before finetuning:")
        print("Test accuracy:", evaluate(model,testloader, DEVICE)) #0.7668
        print("Training accuracy:", evaluate(model, trainloader, DEVICE)) #0.80664

        print("Performance after finetuning:")
        test_accu, train_accu = finetune(model, 10, trainloader, testloader, DEVICE)
        print("Test accuracy:", test_accu)
        print("Training accuracy:", train_accu)
