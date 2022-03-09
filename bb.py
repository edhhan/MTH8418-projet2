from src.FCC_class import FCC
from src.FCC_utils import *


def bb(lr, b1, b2, u1, u2, nb_epoch=25, gpu_flag=True):
    """
    TODO
    :param lr:
    :param b1:
    :param b2:
    :param u1:
    :param u2:
    :param nb_epoch:
    :param gpu_flag:
    :return: a float for the final precision of the trained model on the test set
    """

    # Load training, validation and testing sets formatted with the batch size
    train_loader, validation_loader, test_loader = data_loader(seed=11, batch_size=24)

    # Construct model with specified units in the 1st and 2nd layer
    model = FCC([u1, u2])

    # Decide whether CPU or GPU is used
    gpu_available = torch.cuda.is_available()
    if gpu_available and gpu_flag:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Cast model to proper hardware (CPU or GPU)
    model = model.to(device)

    # Trained model
    trained_model = FCC_main(model, train_loader, validation_loader, nb_epoch, lr, b1, b2, device)

    # Final precision on trained model on test dataset (hold-out set)
    return accuracy(trained_model, test_loader, device)[0]


# Run blackbox on set of hyperparameters
#accuracy = bb(lr=0.1, b1=0.5, b2=0.5, u1=int(5), u2=int(5), nb_epoch=1)
#print(accuracy)
