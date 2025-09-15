from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import torchmetrics
from torchmetrics.functional import pairwise_cosine_similarity


class SoftDiceLoss(nn.Module):
    """
    A soft Dice loss to measure how well differentially expressed genes are captured
    """

    def __init__(
        self,
        temperature: float = 0.5,
        threshold: float = 1,
        variance_parameter: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.temperature = temperature
        self.threshold = threshold
        self.reduction = reduction
        self.variance_parameter = variance_parameter

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        control_exp: torch.Tensor,
        gene_embeddings: torch.Tensor,
        **kwargs,
    ):
        predicted_lfc = (
            torch.abs(y_pred - control_exp) - self.threshold
        ) / self.temperature
        true_lfc = (torch.abs(y_true - control_exp) - self.threshold) / self.temperature

        pred_probs = torch.sigmoid(predicted_lfc)
        true_probs = torch.sigmoid(true_lfc)

        intersection = torch.sum(pred_probs * true_probs, -1)
        # union = pred_probs.sum(-1) + true_probs.sum(-1) - intersection

        calc = 1 - (2 * intersection) / (pred_probs.sum(-1) + true_probs.sum(-1) + 1e-8)

        # unique, indices = gene_embeddings.unique(return_inverse=True, dim=0)
        # loss = torch_scatter.scatter_mean(calc, indices, dim=0) # Gene wise Jaccard averaging
        # std = torch_scatter.scatter_std(calc, indices, dim=0, ) # Buggy - don't use

        # e_loss_2 = torch_scatter.scatter_mean(calc**2, indices, dim=0)

        # variance = e_loss_2 - loss**2 # Var[x] = E[x^2] - (E[x])^2

        final_loss = calc  # + variance * self.variance_parameter

        match self.reduction:
            case "sum":
                return final_loss.sum()
            case "mean":
                return final_loss.mean()
            case _:
                return final_loss


class AdjacencySimilarityLoss(nn.Module):
    """
    This implements the Graph Laplacian loss which acts as a regularizer to ensure smoothness in the data
    """

    def __init__(
        self,
        scaling_type: Literal["linear", "sigmoid", "relu"] = "relu",
        temperature: float = 1,
        threshold: float = 0,
    ):
        """
        Args:
            scaling_type (str): Type of scaling to be performed on the cosine similarity matrix
            temperature (float): Temperature for sigmoid scaling (Defaults to 1)
            threshold (float): Threshold for sigmoid scaling (Defaults to 0)
        """
        super().__init__()

        self.scaling_type = scaling_type
        self.temperature = temperature
        self.threshold = threshold

    def scale_adjacency(self, adjacency: torch.Tensor):
        match self.scaling_type:
            case "linear":
                scaled_adjacency = (1 + adjacency) / 2
            case "sigmoid":
                scaled_adjacency = torch.sigmoid(
                    (adjacency - self.threshold) / self.temperature
                )
            case "relu":
                scaled_adjacency = adjacency.relu()
            case _:
                raise ValueError("Invalid scaling type")

        return scaled_adjacency

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        gene_embeddings: torch.Tensor,
        **kwargs,
    ):
        """
        Calculate Laplacian loss
        """

        shape = gene_embeddings.shape[0]
        gene_adjacency = WeightedContrastiveLoss.cosine_distance(
            gene_embeddings, gene_embeddings
        ) - torch.eye(shape).to(
            device=gene_embeddings.device
        )  # shape bxb and remove diagonal
        gene_adjacency = self.scale_adjacency(gene_adjacency)  # scale between 0 and 1

        pred_adjcacency = WeightedContrastiveLoss.cosine_distance(y_pred, y_pred)
        pred_adjcacency = self.scale_adjacency(pred_adjcacency)

        loss = torch.linalg.norm(gene_adjacency - pred_adjcacency, ord="fro")

        return loss


class LaplacianRegularizerLoss(nn.Module):
    """
    This implements the Graph Laplacian loss which acts as a regularizer to ensure smoothness in the data
    """

    def __init__(
        self,
        scaling_type: Literal["linear", "sigmoid", "relu"] = "relu",
        temperature: float = 1,
        threshold: float = 0,
    ):
        """
        Args:
            scaling_type (str): Type of scaling to be performed on the cosine similarity matrix
            temperature (float): Temperature for sigmoid scaling (Defaults to 1)
            threshold (float): Threshold for sigmoid scaling (Defaults to 0)
        """
        super().__init__()

        self.scaling_type = scaling_type
        self.temperature = temperature
        self.threshold = threshold

    def scale_adjacency(self, adjacency: torch.Tensor):
        match self.scaling_type:
            case "linear":
                scaled_adjacency = (1 + adjacency) / 2
            case "sigmoid":
                scaled_adjacency = torch.sigmoid(
                    (adjacency - self.threshold) / self.temperature
                )
            case "relu":
                scaled_adjacency = adjacency.relu()
            case _:
                raise ValueError("Invalid scaling type")

        return scaled_adjacency

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        gene_embeddings: torch.Tensor,
        **kwargs,
    ):
        """
        Calculate Laplacian loss
        """

        shape = gene_embeddings.shape[0]
        gene_adjacency = WeightedContrastiveLoss.cosine_distance(
            gene_embeddings, gene_embeddings
        ) - torch.eye(shape).to(
            device=gene_embeddings.device
        )  # shape bxb and remove diagonal
        gene_adjacency = self.scale_adjacency(gene_adjacency)  # scale between 0 and 1

        degree_inverse = torch.diag(
            gene_adjacency.sum(-1).clamp(min=1e-8).pow(-0.5)
        )  # shape: b xb

        laplacian = torch.eye(shape).to(device=gene_embeddings.device) - (
            degree_inverse @ gene_adjacency @ degree_inverse
        )

        spec = torch.linalg.norm(laplacian, ord=2)
        scaled_laplacian = laplacian / (spec + 1e-8)

        calc = y_pred.T @ scaled_laplacian @ y_pred  # shape bxb
        loss = torch.trace(calc)

        return loss


class HybridGeneLoss(nn.Module):
    """
    A hybrid loss function that combines a differential expression-aware
    MSE with a soft similarity loss to train on both absolute values
    and relative gene importance.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        temperature: float = 1.0,
        gamma: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            alpha (float): Exponent for MSE weighting. Higher values focus more
                         on the most differentially expressed genes.
            temperature (float): Temperature for the softmax in the similarity loss.
                               Lower values create a sharper distribution.
            gamma (float): A blending factor between 0 and 1. It controls the
                         strength of the two topk component
                         loss = weighted_mse + gamma * similarity_loss
            reduction (str): Reduction to apply to the final loss ('mean' or 'sum').
        """
        super(HybridGeneLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        control_exp: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculate the Loss
        """

        # --- 1. Weighted MSE Component ---
        with torch.no_grad():  # Weights are based on ground truth, no gradient needed
            weights = torch.abs(y_true - control_exp)
            # Min-max scale the weights to [0, 1]
            w_min = torch.min(weights, dim=-1, keepdim=True)[0]
            w_max = torch.max(weights, dim=-1, keepdim=True)[0]
            weights = (weights - w_min) / (w_max - w_min + 1e-8)
            weights = weights**self.alpha

        mse_loss = weights * F.mse_loss(y_pred, y_true, reduction="none")

        # --- 2. Soft Similarity Component ---
        # Calculate LFCs (adding a small epsilon for numerical stability)
        # epsilon = 1e-8
        predicted_lfc = y_pred - control_exp
        true_lfc = y_true - control_exp

        pred_abs = torch.abs(predicted_lfc)
        true_abs = torch.abs(true_lfc)

        pred_weights = F.sigmoid(pred_abs / self.temperature)
        true_weights = F.sigmoid(true_abs / self.temperature)

        similarity = F.cosine_similarity(pred_weights, true_weights, dim=-1)
        similarity_loss = 1 - similarity

        # --- 3. Combine the Losses ---
        # The loss for each item in the batch
        combined_loss = mse_loss.mean(dim=-1) + (self.gamma * similarity_loss)

        # Apply final reduction
        if self.reduction == "mean":
            return combined_loss.mean()
        elif self.reduction == "sum":
            return combined_loss.sum()
        else:
            return combined_loss


class SoftJaccardLoss(nn.Module):
    """
    A soft jaccard loss to measure how well differentially expressed genes are captured
    """

    def __init__(
        self,
        temperature: float = 0.5,
        threshold: float = 1,
        variance_parameter: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.temperature = temperature
        self.threshold = threshold
        self.reduction = reduction
        self.variance_parameter = variance_parameter

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        control_exp: torch.Tensor,
        gene_embeddings: torch.Tensor,
        **kwargs,
    ):
        predicted_lfc = (
            torch.abs(y_pred - control_exp) - self.threshold
        ) / self.temperature
        true_lfc = (torch.abs(y_true - control_exp) - self.threshold) / self.temperature

        pred_probs = torch.sigmoid(predicted_lfc)
        true_probs = torch.sigmoid(true_lfc)

        intersection = torch.sum(pred_probs * true_probs, -1)
        union = pred_probs.sum(-1) + true_probs.sum(-1) - intersection

        calc = 1 - intersection / (union + 1e-8)

        # unique, indices = gene_embeddings.unique(return_inverse=True, dim=0)
        # loss = torch_scatter.scatter_mean(calc, indices, dim=0) # Gene wise Jaccard averaging
        # std = torch_scatter.scatter_std(calc, indices, dim=0, ) # Buggy - don't use

        # e_loss_2 = torch_scatter.scatter_mean(calc**2, indices, dim=0)

        # variance = e_loss_2 - loss**2 # Var[x] = E[x^2] - (E[x])^2

        final_loss = calc  # + variance * self.variance_parameter

        match self.reduction:
            case "sum":
                return final_loss.sum()
            case "mean":
                return final_loss.mean()
            case _:
                return final_loss


class GenewiseMSELoss(nn.Module):
    """
    A macro-averaged MSE loss so that all genes are equally given importance to
    """

    def __init__(self, reduction: str | None = "mean") -> None:
        super(GenewiseMSELoss, self).__init__()

        self.reduction = reduction

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        gene_embeddings: torch.Tensor,
        *args,
        **kwargs,
    ):
        """
        MSE loss but each gene perturbation is treated equally and calculated separately
        and then averaged.
        """

        _, indices = gene_embeddings.unique(return_inverse=True, dim=0)
        diff = F.mse_loss(y_pred, y_true, reduction="none").mean(-1).view(-1)
        calc = torch_scatter.scatter_mean(diff, indices, dim=0)

        match self.reduction:
            case "sum":
                return calc.sum()
            case "mean":
                return calc.mean()
            case _:
                return calc


class MyMSELoss(nn.Module):
    """
    My MSE Loss
    """

    def __init__(self, reduction: str | None = "mean") -> None:
        super(MyMSELoss, self).__init__()

        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
        """
        Generic MSE Loss that will be more compatible with my Composite Loss class

        Args:
             y_pred (Tensor): Predicted expression
             y_true (Tensor): True expression
        Returns:
             Tensor (loss)
        """
        calc = ((y_pred - y_true) ** 2).mean(dim=-1)

        match self.reduction:
            case "sum":
                return calc.sum()
            case "mean":
                return calc.mean()
            case _:
                return calc


class DiffExpAwareMSELoss(nn.Module):
    """
    An MSE loss that focuses on differentially expressed genes with due importance
    to directionality. Adapted from https://www.nature.com/articles/s41587-023-01905-6
    """

    def __init__(
        self,
        temperature: float = 1,
        beta: float = 1,
        threshold: float = 1,
        reduction: str | None = "mean",
    ) -> None:
        super(DiffExpAwareMSELoss, self).__init__()

        # sigmoid part - https://www.desmos.com/calculator/efw5ylni45
        self.beta = beta  # Weight for direction loss
        self.reduction = reduction
        self.temperature = temperature  # Determines sharpness of sigmoid function, weight is determined by
        # how much bigger the fold change is from the threshold
        self.threshold = (
            threshold  # Determines anchor of sigmoid (place where value is 0.5)
        )

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        control_exp: torch.Tensor,
        **kwargs,
    ):
        """
        Calculate the Differential expression aware MSE Loss

        Args:
            y_pred (Tensor): Predicted expression
            y_true (Tensor): True expression
            control (Tensor): Control expression
        """

        # 1. Weighted MSE first
        weights = torch.abs((y_true - control_exp))
        scaled_weights = torch.sigmoid((weights - self.threshold) / self.temperature)
        # weights = (weights - weights.min()) / (
        #     weights.max() - weights.min() + 1e-8
        # ) ** self.alpha
        # weights = weights * weights.mean(dim=-1).pow(-1).view(-1,1) # Normalize with mean
        # weights = weights * 100 # fixed for now, must be a hyperparameter
        # mse_calc = (y_pred - y_true) ** 2
        mse_calc = scaled_weights * (y_pred - y_true) ** 2
        # print(mse_calc)

        # 2. Direction control
        direction_calc = (
            torch.sign((y_true - control_exp)) - torch.sign((y_pred - control_exp))
        ) ** 2

        calc = mse_calc + (direction_calc * self.beta)

        match self.reduction:
            case "sum":
                return calc.sum()
            case "mean":
                return calc.mean()
            case _:
                return calc


class WeightedContrastiveLoss(nn.Module):
    """
    A weighted contrastive loss for ensuring model produces different embeddings for different genetic
    perturbations
    """

    def __init__(
        self,
        temperature: float = 0.1,
        alpha: float = 2.0,
        eps: float = 1e-7,
        gene_hyperbolic=False,
        hyperbolic_similarity_scale=1,
    ):
        super(WeightedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.eps = eps
        self.gene_hyperbolic = gene_hyperbolic
        self.hyperbolic_scale = hyperbolic_similarity_scale

    @staticmethod
    def cosine_distance(u: torch.Tensor, v: torch.Tensor):
        u_norm = F.normalize(u, dim=-1)
        v_norm = F.normalize(v, dim=-1)
        similarity = torch.matmul(u_norm, v_norm.T)

        return similarity

    def _pairwise_poincare_similarity(self, X: torch.Tensor):
        """
        Calculates the pairwise Poincaré similarity between all vectors in a single batch.

        Args:
            X (torch.Tensor): A tensor of shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, batch_size) of distances.
        """
        # Use broadcasting to compute all pairs of differences
        # X_row becomes (batch_size, 1, embedding_dim)
        # X_col becomes (1, batch_size, embedding_dim)
        X_row = X.unsqueeze(1)
        X_col = X.unsqueeze(0)

        sq_dist = torch.sum((X_row - X_col) ** 2, dim=-1)

        # Norms need to be broadcastable as well
        sq_norm = torch.sum(X**2, dim=-1)
        sq_norm_row = sq_norm.unsqueeze(1)
        sq_norm_col = sq_norm.unsqueeze(0)

        # The formula for hyperbolic distance
        numerator = 2 * sq_dist
        denominator = (1 - sq_norm_row) * (1 - sq_norm_col)

        arccosh_arg = 1 + numerator / (denominator + self.eps)
        arccosh_arg = torch.clamp(arccosh_arg, min=1.0 + self.eps)

        distance_matrix = torch.acosh(arccosh_arg)
        similarity_matrix = 2 * torch.exp(-distance_matrix * self.hyperbolic_scale) - 1

        return similarity_matrix

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        gene_embeddings: torch.Tensor,
        **kwargs,
    ):
        """
        Calculate the Contrastive Loss weighted based on genetic similarity

        Args:
            y_pred (Tensor): predicted expression
            y_true (Tensor): not used
            gene_embeddings (Tensor): Emebeddings representing genes
            args, kwargs can be ignored and are present only for consistency

        Returns:
            Tensor (loss)
        """
        batch_size = y_pred.shape[0]
        device = y_pred.device

        # Calculate cosine distances

        pred_sim_mat = (
            self.cosine_distance(y_pred, y_pred) / self.temperature
        )  # Temperature to scale the distances
        if not self.gene_hyperbolic:
            gene_sim_mat = self.cosine_distance(gene_embeddings, gene_embeddings)
        elif self.gene_hyperbolic:
            gene_sim_mat = self._pairwise_poincare_similarity(gene_embeddings)
        else:
            raise ValueError(
                "Invalid value for gene_hyperbolic. Must be boolean. You are idiot"
            )

        # Removing diagonal

        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        pred_sim_off_diag = pred_sim_mat[mask].view(batch_size, batch_size - 1)
        gene_sim_off_diag = gene_sim_mat[mask].view(batch_size, batch_size - 1)

        # Calculating weights
        # positive_gene_sim = F.relu(gene_sim_off_diag)  # Care more about similar genes
        # positive_gene_sim = gene_sim_off_diag

        # Sign is preserved for all powers
        weights = torch.sign(gene_sim_off_diag) * torch.pow(
            gene_sim_off_diag, self.alpha
        )  # Alpha to increase the focus of weights

        exp_pred_sim = torch.exp(pred_sim_off_diag)

        # Push and pull modelling

        pos_mask = (weights > 0).float()
        neg_mask = (weights < 0).float()

        ## positive component: forces similar expression for similar gene - must be increased
        positive_component = (weights * pos_mask * exp_pred_sim).sum(dim=1)
        ## negative component: forces dissimilar expression for dissimilar gene - must be decreased
        negative_component = (torch.abs(weights) * neg_mask * exp_pred_sim).sum(dim=1)

        # log_pos = torch.log(positive_component + self.eps)
        # log_neg = torch.log(negative_component + self.eps)

        score = positive_component / (
            positive_component + negative_component + self.eps
        )
        loss = -torch.log(score).mean()

        # Weighted loss calculation

        # weighted_pos = (weights * exp_pred_sim).sum(dim=1)
        # total_sim = exp_pred_sim.sum(dim=1)

        # if weighted_pos.sum() == 0 or total_sim.sum() == 0:
        #     warn(
        #         "Contrastive loss: One of the distance vectors sums to zero. loss will not be defined"
        #     )

        # loss = -torch.log((weighted_pos + self.eps) / (total_sim + self.eps)).mean()

        return loss


class PerturbationSimilarityLoss(nn.Module):
    """
    Measuring perturbation similarity
    """

    def __init__(self, eps=1e-6, reduction: str | None = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        gene_embeddings: torch.Tensor,
        **kwargs,
    ):
        """
        Calculate the Loss

        Args:
            y_pred (torch.Tensor): Predicted features
            y_true (torch.Tensor): Not used, here for consistency
            gene_embeddings (torch.Tensor): Gene Embeddings

        Returns:
            torch.Tensor (loss)
        """

        triu_mask = torch.triu(
            torch.ones(y_pred.shape[0], y_pred.shape[0], dtype=torch.bool), diagonal=1
        )
        pairwise_similarities = pairwise_cosine_similarity(y_pred)[triu_mask]

        gene_distances = pairwise_cosine_similarity(gene_embeddings)[triu_mask]

        calc = 1 - torchmetrics.functional.spearman_corrcoef(
            pairwise_similarities, gene_distances
        )

        match self.reduction:
            case "sum":
                return calc.sum()
            case "mean":
                return calc.mean()
            case _:
                return calc


class BatchVariance(nn.Module):
    """
    Variance of genes across a batch
    """

    def __init__(self, reduction: str | None = "mean"):
        super(BatchVariance, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
        """
        Calculate the variance of genes across a batch

        Args:
            y_pred (torch.Tensor): Predicted tensor
            y_true (torch.Tensor): Truth tensor (not used)

        Returns:
            torch.Tensor: loss
        """

        calc = y_pred.var(dim=0)

        match self.reduction:
            case "sum":
                return calc.sum()
            case "mean":
                return calc.mean()
            case _:
                return calc


class DiffExpError(nn.Module):
    """
    Measures changes in differential expression between y_pred and y_true
    compared to the control samples
    """

    def __init__(self, reduction: str | None = "mean") -> None:
        super().__init__()

        self.reduction = reduction

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        control_exp: torch.Tensor,
        **kwargs,
    ):
        """
        Calculate the error

        Args:
            y_pred (torch.Tensor): Predicted tensor
            y_true (torch.Tensor): Truth tensor

         Returns:
             torch.Tensor: loss
        """

        diff_true = y_pred - control_exp  # Ensure the expression is log transformed
        diff_pred = y_true - control_exp  # Ensure the expression is log transformed

        calc = 1 - torch.cosine_similarity(
            diff_true, diff_pred
        )  # cosine between the differences

        if self.reduction == "sum":
            return calc.sum()
        elif self.reduction == "mean":
            return calc.mean()
        else:
            return calc


class LogCoshError(nn.Module):
    """
    Log Cosh Loss
    """

    def __init__(self, reduction: str | None = "mean"):
        super(LogCoshError, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
        """
        Calculate the error

        Args:
            y_pred (torch.Tensor): Predicted tensor
            y_true (torch.Tensor): Truth tensor

         Returns:
             torch.Tensor: loss
        """

        calc = torch.log(torch.cosh(y_pred - y_true))

        if self.reduction == "sum":
            return calc.sum()
        elif self.reduction == "mean":
            return calc.mean()
        else:
            return calc


# Caveat: Generated by Gemini and I may not fully understand
class NegativeBinomialLoss(nn.Module):
    """
    Negative Binomial Loss with a learnable dispersion parameter.

    Takes predicted counts (mu) and ground truth counts as input.
    The dispersion parameter `alpha` is a single value learned during training,
    which is suitable when the overdispersion level is constant for the dataset.
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps (float): A small value to add for numerical stability.
        """
        super(NegativeBinomialLoss, self).__init__()
        # Initialize log(alpha) as a learnable parameter.
        # We learn log(alpha) instead of alpha to ensure alpha is always positive.
        # Initializing to 0.0 means alpha starts at 1.0.
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.eps = eps

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Calculates the Negative Binomial loss.

        Args:
            y_pred (torch.Tensor): The predicted counts (mu). Shape: (batch_size, ...).
                                   Must be positive. Models often use a Softplus or
                                   ReLU activation on the final layer to ensure this.
            y_true (torch.Tensor): The ground truth counts. Shape: (batch_size, ...).

        Returns:
            torch.Tensor: The mean loss over the batch.
        """
        # Ensure y_true and y_pred have the same shape
        if y_true.dim() != y_pred.dim():
            y_true = y_true.view_as(y_pred)

        # The predicted values `y_pred` are the mean `mu` of the distribution
        mu = y_pred

        # Get the dispersion parameter `alpha` from the learnable log_alpha
        alpha = torch.exp(self.log_alpha) + self.eps

        # For convenience and stability, let theta = 1 / alpha
        theta = 1.0 / alpha

        # --- Calculate the Negative Log-Likelihood ---
        # lgamma is the log of the Gamma function, used for numerical stability
        log_likelihood = (
            torch.lgamma(y_true + theta)
            - torch.lgamma(theta)
            - torch.lgamma(y_true + 1)
            + theta * torch.log(theta + self.eps)
            + y_true * torch.log(mu + self.eps)
            - (theta + y_true) * torch.log(theta + mu + self.eps)
        )

        # The loss is the negative of the log-likelihood
        loss = -log_likelihood

        # Return the mean loss over the batch
        return loss.mean()


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

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
        """
        Forward pass
        """
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

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
        """
        Forward pass
        """
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

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
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
            final_loss += weight * loss_function.forward(y_pred, y_true, **kwargs)

        return final_loss


# Unclean compositeLosses
#


class MSEandDiffExpLoss(nn.Module):
    """
    A combination of MSE and Differential Expression loss.
    Won't work smoothly in the `CompositeLoss` class because
    the former requires two inputs while the latter requries three
    """

    def __init__(
        self, weights: list[int] = [1, 1], reduction: str | None = "mean"
    ) -> None:
        super().__init__()

        self.weights = weights
        self.mse = torch.nn.MSELoss(reduction=reduction)
        self.diffexp = DiffExpError(reduction=reduction)

    def forward(self, y_pred, y_true, **kwargs):
        """
        Calculate the Loss
        """
        mse_loss = self.mse.forward(y_pred, y_true)
        diffexp = self.diffexp.forward(y_pred, y_true, **kwargs)

        return self.weights[0] * mse_loss + self.weights[1] * diffexp


class MDPLoss(nn.Module):
    """
    A combination of MSE, Diff Exp and Perturbation sensitivity losses.
    Because of the different requirements of each of the losses Composite loss won't work
    Play around with the weights
    """

    def __init__(
        self, weights: list[int] = [1, 1, 1], reduction: str | None = "mean"
    ) -> None:
        super().__init__()

        self.weights = weights
        self.mse = torch.nn.MSELoss(reduction=reduction)
        self.diffexp = DiffExpError(reduction=reduction)
        self.psl = WeightedContrastiveLoss()

    def forward(self, y_pred, y_true, **kwargs):
        """
        Calculate the Loss
        """
        mse_loss = self.mse.forward(y_pred, y_true)
        diffexp = self.diffexp.forward(y_pred, y_true, **kwargs)
        psl = self.psl.forward(y_pred, y_true, **kwargs)

        return (
            (self.weights[0] * mse_loss)
            + (self.weights[1] * diffexp)
            + (self.weights[2] * psl)
        )


if __name__ == "__main__":
    # Verifying composite loss

    loss_fn = CompositeLoss([torch.nn.L1Loss(), WeightedMAELoss(10)], [1, 2])

    preds = torch.randn(5, 10)
    truths = torch.randn(5, 10)

    loss = loss_fn.forward(preds, truths)

    print(f"Loss: {loss:.5f}")
