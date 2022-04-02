from src.FCC_class import *
from src.FCC_utils import *


def bb(o: str, a: str, u1: int, u2: int, r: float, b1: float, b2: float, nb_epoch=25, gpu_flag=True):
    """
    Blackbox function that trains and test the performance of a 2 hidden layer fully-connected for
    a set of hyperparameters
    :param o: the optimizer between "Adam" and "ASGD" (string)
    :param a: the activation function between "ASGD" and "ReLU" (string)
    :param u1: the number of units in the first hidden layer, u1 \in {5,6,.., 49, 50}
    :param u2: the number of units in the second hidden layer, u2 \in {5,6,.., 49, 50}
    :param r: learning rate
    :param b1: 1st hyperparemeter related to the choice of optimizer
    :param b2: 2nd hyperparemeter related to the choice of optimizer
    :param nb_epoch:
    :param gpu_flag:
    :return:
    """

    # Load training, validation and testing sets formatted with the batch size
    train_loader, validation_loader, test_loader = data_loader(seed=11, batch_size=24)

    # Construct model with specified units in the 1st and 2nd layer
    if a == "ReLU":
        model = FCCReLU([u1, u2])
    elif a == "Sigmoid":
        model = FCCSigmoid([u1, u2])
    else:
        raise Exception("Activation must be a string between ReLU or Sigmoid")

    # Decide whether CPU or GPU is used
    gpu_available = torch.cuda.is_available()
    if gpu_available and gpu_flag:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Cast model to proper hardware (CPU or GPU)
    model = model.to(device)

    # Trained model
    hp = (o, r, b1, b2,)
    trained_model = FCC_main(model, train_loader, validation_loader, nb_epoch, device, hp)

    # Final precision on trained model on test dataset (hold-out set)
    return accuracy(trained_model, test_loader, device)[0]


# Run blackbox on set of hyperparameters :
#accuracy = bb2(o="ASGD", a="Sigmoid", u1=int(5), u2=int(5), r=0.001, b1=0.5, b2=0.5, nb_epoch=1)
#print(accuracy)
