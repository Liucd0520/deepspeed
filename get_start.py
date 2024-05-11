import argparse
import deepspeed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
import get_ds_config 
import os
import time 

# run: deepspeed get_start.py 

def add_argument():
    parser = argparse.ArgumentParser(description="CIFAR")

    # For train.
    parser.add_argument(
        "-e",
        "--epochs",
        default=5,
        type=int,
        help="number of total epochs (default: 30)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2000,
        help="output logging information at a given interval",
    )

    # For mixed precision training.
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="Datatype used for training",
    )

    # For ZeRO Optimization.
    parser.add_argument(
        "--stage",
        default=2,
        type=int,
        choices=[0,  1, 2, 3],
        help="Datatype used for training",
    )

    # For MoE (Mixture of Experts).
    parser.add_argument(
        "--moe",
        default=False,
        action="store_true",
        help="use deepspeed mixture of experts (moe)",
    )
    parser.add_argument(
        "--ep-world-size", default=1, type=int, help="(moe) expert parallel world size"
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        nargs="+",
        default=[
            1,
        ],
        help="number of experts list, MoE related.",
    )
    parser.add_argument(
        "--mlp-type",
        type=str,
        default="standard",
        help="Only applicable when num-experts > 1, accepts [standard, residual]",
    )
    parser.add_argument(
        "--top-k", default=1, type=int, help="(moe) gating top 1 and 2 supported"
    )
    parser.add_argument(
        "--min-capacity",
        default=0,
        type=int,
        help="(moe) minimum capacity of an expert regardless of the capacity_factor",
    )
    parser.add_argument(
        "--noisy-gate-policy",
        default=None,
        type=str,
        help="(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter",
    )
    parser.add_argument(
        "--moe-param-group",
        default=False,
        action="store_true",
        help="(moe) create separate moe param groups, required when using ZeRO w. MoE",
    )

    # Include DeepSpeed configuration arguments.
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args




class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)

        return x


# Initialize DeepSpeed distributed backend.
deepspeed.init_distributed()

transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

if torch.distributed.get_rank() != 0:
        # Might be downloading cifar data, let rank 0 download first.
        torch.distributed.barrier()

 # Load or download cifar data.
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

if torch.distributed.get_rank() == 0:
        # Cifar data is downloaded, indicate other ranks can proceed.
        torch.distributed.barrier()

net = Net()

 # Get list of parameters that require gradients.
parameters = filter(lambda p: p.requires_grad, net.parameters())

args = add_argument()  
ds_config = get_ds_config.process_config(args=args)

model_engine, optimizer, trainloader, _ = deepspeed.initialize(
     args=args, 
     model=net, 
     model_parameters=parameters,
     training_data=trainset,
     config=ds_config
)

# Get the local device name (str) and local rank (int).
local_device = get_accelerator().device_name(model_engine.local_rank)
local_rank = model_engine.local_rank

print('-----x-------------------------')
start = time.time()

target_dtype = None

if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
elif model_engine.fp16_enabled():
    target_dtype = torch.half

# Define the Classification Cross-Entropy loss function.
criterion = nn.CrossEntropyLoss()

for epoch in range(args.epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # Get the inputs. ``data`` is a list of [inputs, labels].
        inputs, labels = data[0].to(local_device), data[1].to(local_device)

        # Try to convert to target_dtype if needed.
        if target_dtype != None:
            inputs = inputs.to(target_dtype)

        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()
       
        # Print statistics
        running_loss += loss.item()
        if local_rank == 0 and i % args.log_interval == (
            args.log_interval - 1
        ):  # Print every log_interval mini-batches.
            print(
                f"[{epoch + 1 : d}, {i + 1 : 5d}] loss: {running_loss / args.log_interval : .3f}"
            )
            running_loss = 0.0

print("Finished Training")
print(time.time() - start)  


def test(model_engine, testset, local_device, target_dtype, test_batch_size=4):
    """Test the network on the test data.

    Args:
        model_engine (deepspeed.runtime.engine.DeepSpeedEngine): the DeepSpeed engine.
        testset (torch.utils.data.Dataset): the test dataset.
        local_device (str): the local device name.
        target_dtype (torch.dtype): the target datatype for the test data.
        test_batch_size (int): the test batch size.

    """
    # The 10 classes for CIFAR10.
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # Define the test dataloader.
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=0
    )

    # For total accuracy.
    correct, total = 0, 0
    # For accuracy per class.
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    # Start testing.
    model_engine.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if target_dtype != None:
                images = images.to(target_dtype)
            outputs = model_engine(images.to(local_device))
            _, predicted = torch.max(outputs.data, 1)
            # Count the total accuracy.
            total += labels.size(0)
            correct += (predicted == labels.to(local_device)).sum().item()

            # Count the accuracy per class.
            batch_correct = (predicted == labels.to(local_device)).squeeze()
            for i in range(test_batch_size):
                label = labels[i]
                class_correct[label] += batch_correct[i].item()
                class_total[label] += 1

    if model_engine.local_rank == 0:
        print(
            f"Accuracy of the network on the {total} test images: {100 * correct / total : .0f} %"
        )

        # For all classes, print the accuracy.
        for i in range(10):
            print(
                f"Accuracy of {classes[i] : >5s} : {100 * class_correct[i] / class_total[i] : 2.0f} %"
            )



########################################################################
# Step 4. Test the network on the test data.
########################################################################
test(model_engine, testset, local_device, target_dtype)
print('xxxxx')



# epoch 5  batch_size 160
# stage   Memory     Time
# 0       1120        9.2
# 1       1122        9       
# 2       1316        11
# 3       1220        24
