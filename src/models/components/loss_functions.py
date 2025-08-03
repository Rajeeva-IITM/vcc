import torch
import torch.nn as nn
import torchmetrics


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
        # Normalizing weights to the range of 0 to 1
        # positive_weights = (self.weights - self.weights.min()) / (self.weights.max() - self.weights.min())

        positive_weights = torch.nn.functional.softplus(self.weights)

        weighted_error = abs_error * positive_weights.unsqueeze(
            0
        )  # broadcast across batch
        loss = weighted_error.mean()

        return loss


class CompositeLoss(nn.Module):
    """CompositeLoss combines Mean Absolute Error (MAE) and Cosine losses
    into a single loss function for regression tasks.

    Arguments
    ---------
    lambda_mae : float
        Weight for the MAE loss component.
    lambda_mse : float
        Weight for the cosine loss component.

    Attributes
    -----------
    lambda_val : float
        Weight for the losses.

    Methods
    -------
    forward(y_pred, y_true)
        Computes the weighted sum of MAE and (1 - Pearson cosine coefficient) as the loss.
    """

    def __init__(self, lambda_val: float) -> None:
        super(CompositeLoss, self).__init__()

        assert 0 <= lambda_val <= 1, "lambda value must be between 0 and 1"

        self.lambda_val = lambda_val
        # self.mae_loss = torchmetrics.MeanAbsoluteError()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Computes a custom loss as a weighted sum of MAE loss and (1 - Pearson correlation coefficient).
        Args:
            y_pred (Tensor): Predicted values.
            y_true (Tensor): Ground truth values.
        Returns:
            Tensor: Computed loss value.
        """

        mae = torchmetrics.functional.mean_absolute_error(y_true, y_pred)
        cosine = torchmetrics.functional.cosine_similarity(
            y_true, y_pred, reduction="mean"
        )

        loss = self.lambda_val * mae + (1 - self.lambda_val) * (1 - cosine)

        return loss
