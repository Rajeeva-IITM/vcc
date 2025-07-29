import warnings

from torch import nn


def get_activation(name: str):
    """Returns the activation function based on the string

    Parameters
    ----------
    name : str
        Name of the activation function
    """

    # Make case insensitive

    name = name.lower()

    match name:
        case "relu":
            return nn.ReLU()
        case "tanh":
            return nn.Tanh()
        case "sigmoid":
            return nn.Sigmoid()
        case "celu":
            return nn.CELU()
        case "gelu":
            return nn.GELU()
        case _:
            warnings.warn(
                """String doesn't match one of ['relu','tanh','sigmoid','celu','gelu'].
                Proceeding with default Relu"""
            )
            return nn.ReLU()
