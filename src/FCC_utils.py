import random
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim


def data_loader(seed=11, batch_size=24):
    """
    TODO
    :param seed:
    :param batch_size:
    :return:
    """
    # Set seed for reproducible results
    random.seed(seed)
    torch.manual_seed(seed)

    # FashionMNIST training and validation dataset
    train_valid_data = datasets.FashionMNIST('./data',
                                             train=True,
                                             download=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5,), (0.5,))]))

    # Randomly draw 12k data, where 10k for training and 2k for validation
    train_valid_data, discarded_data = torch.utils.data.random_split(train_valid_data, [10000, 50000])
    train_data, valid_data = torch.utils.data.random_split(train_valid_data, [8000, 2000])

    # Randomly draw 3k for the test dataset
    test_data = datasets.FashionMNIST('./data',
                                      train=False,
                                      download=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))]))
    test_data, discarded_data = torch.utils.data.random_split(test_data, [2000, 8000])
    del discarded_data

    # Prepare data loaders for training, validation and testing datasets
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader, test_loader


# Training for a single epoch
def train(model, loader, optimizer, device):
    """
    TODO
    :param model:
    :param loader:
    :param optimizer:
    :param device:
    :return:
    """
    model.train()
    loss_training = 0

    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        loss_training = loss_training + loss.item()

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


# Accuracy on training, validation or test set
def accuracy(model, loader, device):
    """
    TODO
    :param model:
    :param loader:
    :param device:
    :return:
    """
    # Disable graph and gradient computations for speed and memory efficiency
    with torch.no_grad():
        model.eval()
        loss, nb_correct = 0.0, 0
        nb_of_batches = len(loader.dataset)

        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            predictions = outputs.data.max(1, keepdim=True)[1]

            # Losses and nb of correct predictions
            loss += F.nll_loss(outputs, targets, reduction='sum').item()
            nb_correct += predictions.eq(targets.data.view_as(predictions)).cpu().sum()

    return float(nb_correct / nb_of_batches) * 100, loss / nb_of_batches


# Train on all epochs and select the best trained model based on accuracy on the validation set
def FCC_main(model, train_loader, validation_loader,  epochs, device, hp):
    """
    Main
    :param model:
    :param train_loader:
    :param validation_loader:
    :param epochs:
    :param device:
    :param hp:
    :return:
    """
    # Unwrap hyperparameters contained in list hp = (o, a, u1, u2, r, b1, b2,)
    o = hp[0]
    lr = hp[1]
    b1 = hp[2]
    b2 = hp[3]

    if o == "Adam":
        optimizer = optim.Adam(model.parameters(), betas=(b1, b2), lr=lr, weight_decay=0.05)
    elif o == "ASGD":
        optimizer = optim.ASGD(model.parameters(), lr=lr, lambd=b2, alpha=b1)
    else:
        raise Exception("Optimizer must be a string between Adam or ASGD")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    best_precision = 0
    training_losses = []
    accuracies = []
    validation_losses = []

    for epoch in range(epochs):

        # Train
        model = train(model, train_loader, optimizer, device)

        # Accuracy on training set
        train_precision, train_loss = accuracy(model, train_loader, device)
        training_losses.append(train_loss)

        # Accuracy on validation set
        precision, validation_loss = accuracy(model, validation_loader, device)
        validation_losses.append(validation_loss)
        accuracies.append(precision)

        if precision > best_precision:
            best_precision = precision
            best_model = model

        # Scheduler
        scheduler.step(validation_loss)

    return best_model


