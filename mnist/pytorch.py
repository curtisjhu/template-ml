import torch
import torch.nn as nn
from torch.utils.data import Dataset, sampler, DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
from torchsummary import summary
import torchvision.transforms as transforms


###### SETUP ########
dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device: ", device)

#### DATASET #####
batch_size = 32
mnist_train = datasets.MNIST(".", download=True, train=True, transform=transforms.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(50000)))

mnist_vals = datasets.MNIST('.', download = True, train = True, transform = transforms.ToTensor())
loader_vals = DataLoader(mnist_vals, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(50000, 60000)))


##### TRAINING LOGIC #####
def train(model, optimizer, loader_train, loader_val, epochs=1, print_every=100):
    """
    Train a model on MNIST using the PyTorch Module API and prints model
    accuracies during training.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - loader_train: Dataloader for training
    - loader_val: Dataloader for evaluation
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    - print_every: Number of iterations at which the accuracy of the model
      should be evaluated periodically

    Returns: Lists of validation accuracies at the end of each epoch.
    """
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    train_accs = []
    val_accs = []
    for e in range(epochs):
        print('-' * 128)
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = loss_fn(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each trainable parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(loader_val, model)
                print()
        val_accs.append(check_accuracy(loader_val, model))
    return val_accs


def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc




##################################################################
##################################################################
###### ##### ## ##################################################
######  ###  ## ##################################################
###### # # # ## ##################################################
###### ## ## ## ##################################################
###### ##### ## ##################################################
##################################################################
##################################################################

model = nn.Sequential(
    # (N, 1, 28, 28)
    nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
    ),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(16, 32, 5, 1, 2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 10),
)

optimizer = optim.Adam(model.parameters(), lr = 0.002)

###### DETAILS ON MODEL #######

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Parameters", count_parameters(model))
summary(model, (1, 28, 28))


######## TRAIN ########

train(model, optimizer, loader_train, loader_vals, epochs=5, print_every=200)

torch.save(model.state_dict(), "pytorch-model.pt")

