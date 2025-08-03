import torch
import torch.nn as nn
import torchmetrics


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
