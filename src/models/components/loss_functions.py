import torch
import torch.nn as nn

# import torchmetrics


class MSLE(nn.Module):
    """
    Mean Squared Logarithmic Error (MSLE) loss module.

    This loss function computes the mean squared logarithmic error between the predicted and true values.
    It is particularly useful when targets can span several orders of magnitude and penalizes underestimates more than overestimates.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            'mean' | 'sum' | None. 'mean': the sum of the output will be divided by the number of elements in the output.
            'sum': the output will be summed. None: no reduction will be applied. Default: 'mean'.

    Shape:
        - y_pred: (N, *) where * means any number of additional dimensions
        - y_true: (N, *), same shape as y_pred

    Returns:
        torch.Tensor: The calculated MSLE loss. If reduction is 'none', returns the unreduced loss with the same shape as input.
        Otherwise, returns a scalar.

    Example:
        >>> criterion = MSLE(reduction='mean')
        >>> y_pred = torch.tensor([2.5, 0.0, 2.0, 7.0])
        >>> y_true = torch.tensor([3.0, 0.0, 2.0, 8.0])
        >>> loss = criterion(y_pred, y_true)

    Mean Squared Log Error loss.
    """

    def __init__(self, reduction: str | None = "mean") -> None:
        super(MSLE, self).__init__()

        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        calc = (torch.log1p(y_true) - torch.log1p(y_pred)) ** 2

        if self.reduction == "sum":
            return calc.sum()
        elif self.reduction == "mean":
            return calc.mean()
        else:
            return calc


class WeightedMAELoss(nn.Module):
    """
    A weighted MAE loss. The weights are learned by the model to prioritize specific genes

    Arguments
    ---------
    num_genes (int)
         Total number of genes/features in the model
    init_weights Optional(Tensor)
         Initial weights, if none initialized as a vector of ones

    Attributes
    -----------
    weights: (Tensor)
         Weights of the genes

    Methods
    -------
    forward(y_pred, y_true)
        Forward pass through model

    """

    def __init__(
        self, num_genes: int, init_weights: torch.Tensor | None = None
    ) -> None:
        super(WeightedMAELoss, self).__init__()

        if init_weights is None:
            # Initialize weights to one
            init_weights = torch.ones(num_genes)
        self.weights = nn.Parameter(init_weights)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        abs_error = torch.abs(y_true - y_pred)
        # Ensure weights are on the same device as y_pred
        weights = self.weights.to(y_pred.device)
        positive_weights = torch.nn.functional.softplus(weights)

        weighted_error = abs_error * positive_weights.unsqueeze(
            0
        )  # broadcast across batch
        loss = weighted_error.mean()

        return loss


class CompositeLoss(nn.Module):
    """CompositeLoss combines multiple losses in a weighted fashion

    Arguments
    ---------
    loss_functions: List[nn.Module]
        List of loss functions (must have the `forward` method implemented)
    weights:
        Weights to be given to each loss function

    Attributes
    -----------
    loss_functions: list[nn.Module]
        List of individual losses
    weights: list[int | float]


    Methods
    -------
    forward(y_pred, y_true)
        Computes the weighted sum of individual losses.
    """

    def __init__(
        self, loss_functions: list[nn.Module], weights: list[int | float]
    ) -> None:
        super(CompositeLoss, self).__init__()

        assert len(loss_functions) == len(weights), (
            "Number of weights must be equal to the number of loss functions provided"
        )
        self.loss_functions = loss_functions
        self.weights = weights

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Computes a custom loss as a weighted sum of the constituent losses.
        Args:
            y_pred (Tensor): Predicted values.
            y_true (Tensor): Ground truth values.
        Returns:
            Tensor: Computed loss value.
        """

        final_loss = 0

        for loss_function, weight in zip(self.loss_functions, self.weights):
            final_loss += weight * loss_function.forward(y_pred, y_true)

        return final_loss


if __name__ == "__main__":
    # Verifying composite loss

    loss_fn = CompositeLoss([torch.nn.L1Loss(), WeightedMAELoss(10)], [1, 2])

    preds = torch.randn(5, 10)
    truths = torch.randn(5, 10)

    loss = loss_fn.forward(preds, truths)

    print(f"Loss: {loss:.5f}")
