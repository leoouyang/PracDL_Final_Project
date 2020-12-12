from train_AlexNet import *
from prune_structured_AlexNet import *

import torch
import torch.nn as nn
import torchvision.models as models

import torch_pruning as pruning


def get_average_inference_time(model, device, batch_size_list):
    model.eval()
    with torch.no_grad():
        # do a inference to make sure the model is loaded in GPU
        output = model(torch.randn(1, 3, 224, 224).to(device))

        result = []
        for batch_size in batch_size_list:
            inference_time_sum = 0
            for i in range(10):
                x = torch.randn(batch_size, 3, 224, 224).to(device)

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = model(x)
                end.record()
                torch.cuda.synchronize()

                cur_inference_time = start.elapsed_time(end)
                inference_time_sum += cur_inference_time
            result.append(inference_time_sum/10)

    return result

if __name__ == "__main__":
    batch_size = 128
    # For testing inference time
    batch_size_list = [1, 4, 16, 64, 256]
    DEVICE = "cuda"
    MODEL_PATH = './alexnet_finetuned.pth'
    FRAC = "53.14"
    PRUNED_MODEL_PATH = "alexnet_chkpt/model_conv_frac_%s.pth"%FRAC

    trainset, testset = load_cifar10_pytorch(root='G:\ML dataset',
                                             transform=ImageNet_Transform_Func)
    # trainset, testset = load_cifar10_pytorch(transform=ImageNet_Transform_Func)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    inference_time_matrix = [batch_size_list]
    num_params_list = []
    print("==================== Original Model ====================")
    model = models.alexnet()
    model.classifier[6] = nn.Linear(4096, 10)
    model.to(DEVICE)
    load_chkpt(model, MODEL_PATH, DEVICE)

    conv_modules = get_conv_modules(model.features)
    num_params = get_num_params(conv_modules)
    print("Number of params:", num_params)
    num_params_list.append(num_params)
    print("Test Accuracy", evaluate(model, testloader, DEVICE))

    average_inference_time = get_average_inference_time(model, DEVICE, batch_size_list)
    print("Batch size list for evaluating inference time:", batch_size_list)
    print("Average Inference time in ms:", average_inference_time)
    inference_time_matrix.append(average_inference_time)
    print("==================== Model Pruned with Pytorch (with mask) ====================")
    model_pruned = torch.load(PRUNED_MODEL_PATH)
    model_pruned.to(DEVICE)
    conv_modules = get_conv_modules(model_pruned.features)
    print("Number of params:", get_num_params(conv_modules))
    # Here we use mask sum as number of parameters since this is the number of
    # effective parameters pytorch think there it is.
    num_params = print_mask_sum(model_pruned.features)
    num_params_list.append(num_params)
    print("Weight shape in each convolutional layer")
    for m in conv_modules:
        print(m.weight.shape)
    print("Test Accuracy", evaluate(model_pruned, testloader, DEVICE))

    average_inference_time = get_average_inference_time(model_pruned, DEVICE, batch_size_list)
    print("Batch size list for evaluating inference time:", batch_size_list)
    print("Average Inference time in ms:", average_inference_time)
    inference_time_matrix.append(average_inference_time)

    # This only make the prune permanant, the size of the model remains the same
    # The pruned weight are set to be zero. This won't affect the inference time
    # This is required to make the pruning package work.
    for module in conv_modules:
        prune.remove(module, 'weight')
    print("Number of params after removing mask:", get_num_params(conv_modules))
    for module in conv_modules:
        filter_sum_weight = module.weight.sum(dim=(1,2,3)).detach().cpu().numpy()
        pruning_idxs = np.where(filter_sum_weight==0)[0].tolist()
        # print(pruning_idxs)
        DG = pruning.DependencyGraph()
        DG.build_dependency(model_pruned, example_inputs=torch.randn(1, 3, 224, 224))
        pruning_plan = DG.get_pruning_plan(module, pruning.prune_conv, idxs = pruning_idxs)
        pruning_plan.exec()

    print("==================== Actual pruned Model ====================")
    num_params = get_num_params(conv_modules)
    print("Number of params:", num_params)
    num_params_list.append(num_params)
    print("Weight shape in each convolutional layer")
    for m in conv_modules:
        print(m.weight.shape)

    model_pruned.to(DEVICE)
    print("Actual Pruned Test Accuracy before finetuning:", evaluate(model_pruned, testloader, DEVICE))
    test_accu, train_accu = finetune(model_pruned, 5, trainloader, testloader, DEVICE)
    print("Actual Pruned Test Accuracy before finetuning:", test_accu)

    average_inference_time = get_average_inference_time(model_pruned, DEVICE, batch_size_list)
    print("Batch size list for evaluating inference time:", batch_size_list)
    print("Average Inference time in ms:", average_inference_time)
    inference_time_matrix.append(average_inference_time)

    np.savetxt("performance/Alexnet_structured_%s_efficiency.txt"%FRAC, inference_time_matrix)
    np.savetxt("performance/Alexnet_structured_%s_num_params.txt"%FRAC, num_params_list)
