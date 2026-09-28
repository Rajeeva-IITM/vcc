"""Loss terms and regularizers agnostic to whether the target is expression or velocity.

These operate on ``y_pred``/``y_true`` directly (plain MSE, log-cosh, variance), on the
gene-embedding geometry (contrastive / Laplacian / adjacency regularizers), or wrap other
losses (``CompositeLoss``). Space-specific losses live in ``expression`` and ``velocity``.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


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
        teacher: str = "gene",
    ):
        super(WeightedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.eps = eps
        self.gene_hyperbolic = gene_hyperbolic
        self.hyperbolic_scale = hyperbolic_similarity_scale
        # Which pairwise-similarity TEACHER drives the pull/push weights:
        #   "gene"          -- cosine of `gene_embeddings` (the frozen `ko_vec`). Back-compat
        #                      default. On the trio concat this sits in a positive cone
        #                      (mean cosine ~0.26, <1% negative pairs) so the push term almost
        #                      never fires and the loss collapses to ~1e-4 -- see `teacher`
        #                      below for why that is inert.
        #   "gene_centered" -- the same cosine minus its batch-mean, so ~half the pairs become
        #                      negative (push). Makes the embedding teacher fire, but it still
        #                      supervises toward embedding geometry (CKA ~0.11 with response).
        #   "response"      -- cosine of the batch-centered TRUE DELTA `y_true = u = x1 - x0`.
        #                      Removing the batch mean strips the common shift (which dominates
        #                      ~7x and would recreate the positive cone), leaving the
        #                      perturbation-SPECIFIC direction -- the same quantity `specific_cos`
        #                      scores, and the response geometry itself. The student is centered
        #                      the same way so both live in the specific-direction space.
        if teacher not in ("gene", "gene_centered", "response"):
            raise ValueError(
                f"teacher must be 'gene', 'gene_centered' or 'response', got {teacher!r}"
            )
        self.teacher = teacher

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
        Calculate the Contrastive Loss weighted by a pairwise-similarity teacher.

        Args:
            y_pred (Tensor): predicted delta/velocity (the student).
            y_true (Tensor): true delta ``u = x1 - x0``. Used only when
                ``teacher == "response"``; ignored otherwise.
            gene_embeddings (Tensor): per-sample embedding (the frozen ``ko_vec``). Used by
                the ``"gene"`` / ``"gene_centered"`` teachers; ignored for ``"response"``.
            args, kwargs can be ignored and are present only for consistency

        Returns:
            Tensor (loss)
        """
        batch_size = y_pred.shape[0]
        device = y_pred.device

        # float32 so the cosine/exp/log chain keeps precision under bf16-mixed.
        y_pred = y_pred.float()

        # Student similarity. For the response teacher, compare the SPECIFIC directions of
        # the predictions (remove the batch common shift) so the student lives in the same
        # space as the centered-delta teacher; otherwise use the raw predicted profile.
        pred = (
            y_pred - y_pred.mean(dim=0, keepdim=True)
            if self.teacher == "response"
            else y_pred
        )
        pred_sim_mat = (
            self.cosine_distance(pred, pred) / self.temperature
        )  # Temperature to scale the distances

        # Teacher similarity -> the pull/push weights.
        if self.teacher == "response":
            # Batch-center the true delta to strip the common shift, then cosine: the
            # perturbation-specific response geometry, with genuine negatives.
            tgt = y_true.float() - y_true.float().mean(dim=0, keepdim=True)
            gene_sim_mat = self.cosine_distance(tgt, tgt)
        elif self.gene_hyperbolic:
            gene_sim_mat = self._pairwise_poincare_similarity(gene_embeddings.float())
        else:
            gene_sim_mat = self.cosine_distance(
                gene_embeddings.float(), gene_embeddings.float()
            )
            if self.teacher == "gene_centered":
                # A positive-cone embedding has almost no negative pairs; subtract the
                # off-diagonal mean so ~half the pairs push and the loss is informative.
                off = gene_sim_mat[
                    ~torch.eye(batch_size, dtype=torch.bool, device=device)
                ]
                gene_sim_mat = gene_sim_mat - off.mean()

        # Removing diagonal

        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        pred_sim_off_diag = pred_sim_mat[mask].view(batch_size, batch_size - 1)
        gene_sim_off_diag = gene_sim_mat[mask].view(batch_size, batch_size - 1)

        # Calculating weights
        # positive_gene_sim = F.relu(gene_sim_off_diag)  # Care more about similar genes
        # positive_gene_sim = gene_sim_off_diag

        # Sign is preserved for all powers. Take the power of the magnitude:
        # torch.pow on a negative base is NaN for any non-integer alpha.
        weights = torch.sign(gene_sim_off_diag) * gene_sim_off_diag.abs().pow(
            self.alpha
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

        # eps in BOTH numerator and denominator. An anchor whose perturbation has no
        # embedding-similar neighbour in the batch has positive_component == 0; without the
        # numerator eps that is score == 0 and -log(0) == +inf, which NaNs the run the first
        # time such a batch appears (rare per step, near-certain over a full run). With it, a
        # similar-neighbour-less anchor whose only neighbours are dissimilar is still penalised
        # (small score -> large finite loss), and an anchor with neither similar nor dissimilar
        # neighbours (all cosines ~0) scores ~1 -> ~0 loss, i.e. uninformative, as it should be.
        score = (positive_component + self.eps) / (
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
        # ModuleList, not a plain list: otherwise child losses are invisible to
        # `.parameters()` and `.to(device)`, so any loss holding a parameter or
        # buffer silently never trains and stays on the wrong device.
        self.loss_functions = nn.ModuleList(loss_functions)
        self.weights = weights

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
        """
        Computes a custom loss as a weighted sum of the constituent losses.
        Args:
            y_pred (Tensor): Predicted values.
            y_true (Tensor): Ground truth values.
        Returns:
            Tensor: Computed loss value.

        Side effect: the *unweighted* value of each constituent loss from this call is
        stashed in ``self.last_components`` (a ``{class_name: detached_scalar}`` dict) so a
        LightningModule can log each term separately without this loss needing logger access.
        The unweighted value is the diagnostic scale -- independent of the ``weights`` sweep,
        it says whether a term is actually contributing. Duplicate class names are
        disambiguated by position. Detached, so the stash never holds the graph.
        """

        final_loss = 0
        components: dict[str, torch.Tensor] = {}

        for i, (loss_function, weight) in enumerate(
            zip(self.loss_functions, self.weights)
        ):
            term = loss_function.forward(y_pred, y_true, **kwargs)

            name = type(loss_function).__name__
            if name in components:
                name = f"{name}_{i}"
            components[name] = term.detach()

            final_loss = final_loss + weight * term

        self.last_components = components

        return final_loss


class GateSparsityLoss(nn.Module):
    """Pull the decoder's DEG gate toward a target activation rate.

    ``GatedProcessingNN`` predicts ``y = control + gate * (raw - control)``, where
    ``gate`` is a per-gene sigmoid. Nothing else in the objective references it,
    and left free it drifts toward 1.0 -- measured at 0.87 by epoch 10 of a plain
    MSE run -- which makes the expression collapse to ``y = raw`` and the mask
    inert.

    This penalises the *mean* activation against a target rate, as the KL between
    two Bernoullis::

        KL(rho || rho_hat) = rho*log(rho/rho_hat) + (1-rho)*log((1-rho)/(1-rho_hat))

    Constraining the mean rather than each element is deliberate: an elementwise
    penalty drives every gate to ``rho``, which is uniform and masks nothing. The
    mean constraint leaves the model free to open a few genes fully and close the
    rest, and the data term decides which. The uniform solution minimises this
    term alone but not the total loss, since ``gate == rho`` everywhere makes
    ``y ~ control`` and the reconstruction term punishes the genes that do move.

    ``mode="l1"`` is the plain sparsity alternative. It only pushes down, so an
    over-large weight closes the gate entirely and degenerates to ``y = control``;
    the KL is two-sided and targets a rate, which makes the weight less delicate.

    Parameters
    ----------
    target_rate : float
        Fraction of genes expected to deviate from control. Default 0.023, the
        rate measured by Mann-Whitney across all 300 perturbations of the 2025
        data at FDR < 0.05 and |log2FC| >= 0.25.
    mode : {"kl", "l1"}
        Two-sided KL toward ``target_rate``, or one-sided L1 toward zero.
    per_cell : bool
        Average the gate over genes within each cell (default) -- "each cell
        deviates from its control in ~target_rate of its genes". When False,
        average over the batch per gene instead -- "each gene moves in
        ~target_rate of cells".
    eps : float
        Clamp keeping the log arguments away from 0 and 1.
    """

    def __init__(
        self,
        target_rate: float = 0.023,
        mode: str = "kl",
        per_cell: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if not 0.0 < target_rate < 1.0:
            raise ValueError(f"target_rate must be in (0, 1), got {target_rate}")
        if mode not in ("kl", "l1"):
            raise ValueError(f"mode must be 'kl' or 'l1', got {mode!r}")

        self.target_rate = target_rate
        self.mode = mode
        self.per_cell = per_cell
        self.eps = eps

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        gate: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Zero when the net has no gate, so this is safe in any CompositeLoss."""
        if gate is None:
            return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)

        if self.mode == "l1":
            return gate.abs().mean()

        # Mean activation: over genes within a cell, or over cells within a gene.
        rho_hat = gate.mean(dim=-1) if self.per_cell else gate.mean(dim=0)
        rho_hat = rho_hat.clamp(self.eps, 1.0 - self.eps)
        rho = self.target_rate

        kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log(
            (1 - rho) / (1 - rho_hat)
        )
        return kl.mean()


class BatchLaplacianReg(nn.Module):
    """Batch-only graph-Laplacian (Dirichlet) smoothness on predictions.

    A memory-efficient, velocity-correct sibling of :class:`LaplacianRegularizerLoss`, which
    materialises an ``18080 x 18080`` matrix via ``y_pred.T @ L @ y_pred``. This computes the
    identical Dirichlet energy through the ``B x B`` Gram matrix instead, so cost scales with the
    batch, not the gene panel. Batch-only: it reads just ``y_pred`` and ``gene_embeddings``
    (the ``ko_vec`` already in every batch) -- no ``control_exp``, no ``ko_id``, no prep files.

    What it penalises
    -----------------
    For predictions ``y`` and a gene-embedding affinity ``W`` (cosine similarity, zero diagonal)::

        E = sum_ij W_ij * ||y_i - y_j||^2

    The energy is large when two perturbations CLOSE in embedding space have DIFFERENT predicted
    responses. Minimising it pulls each perturbation's prediction toward its embedding-neighbours
    -- partial pooling, a random-effect smoothness prior. Unlike weight decay, which shrinks the
    perturbation effect toward zero (= the perturbation-blind mean), this shrinks toward the
    neighbour-predicted response, which is the quantity that transfers to held-out genes.

    Why the same-gene pairs are kept
    --------------------------------
    Cells sharing a perturbation carry identical embeddings, so their affinity is ~1 and the term
    penalises any difference in their predicted velocity. Under ``target_mode="perturbation_mean"``
    the target ``u = mu_p - x_0`` is identical for those cells, so the penalty is zero at the
    optimum while enforcing ``v(x_t, t) = v(x_t', t')`` across the flow path. Under
    ``target_mode="cell"`` (the datamodule default) it additionally smooths per-cell sampling
    noise, which is benign.

    Efficient form
    --------------
    With ``S = y @ y.T`` (the ``B x B`` Gram), ``sq = diag(S) = ||y_i||^2`` and ``deg = W.sum(1)``::

        sum_ij W_ij (||y_i||^2 + ||y_j||^2 - 2 y_i.y_j) = 2 * (deg . sq) - 2 * sum(W * S)

    equal to ``trace(y.T (D - W) y)`` at ``B x B`` cost. The result is divided by the total edge
    weight ``W.sum()`` (mean squared distance per edge) and by the gene count ``G``, so it sits on
    the same scale as :class:`MyMSELoss` and a ``CompositeLoss`` weight keeps its meaning.

    Parameters
    ----------
    scaling_type : {"relu", "sigmoid", "linear"}
        How the raw cosine similarity maps to a non-negative affinity, mirroring
        :class:`LaplacianRegularizerLoss`. ``relu`` keeps only positive similarities as edges.
    temperature, threshold : float
        Sigmoid parameters, used only when ``scaling_type="sigmoid"``.
    gene_hyperbolic : bool
        Use the Poincare similarity of :class:`WeightedContrastiveLoss` instead of cosine. Off by
        default; the log-mapped embeddings this was tuned on use cosine.
    eps : float
        Denominator floor.
    reduction : str | None
        Present for API parity; the energy is already a scalar, so every value returns it as-is.

    Notes
    -----
    ``gene_mask`` is accepted and ignored here: correct for single-panel runs (the common case,
    where no mask is allocated). Multi-panel correctness -- restricting each pairwise distance to
    the genes both sources measured -- is deferred, exactly as :class:`BatchDEAwareMSELoss` defers
    its own ``gene_mask`` handling.
    """

    def __init__(
        self,
        scaling_type: Literal["relu", "sigmoid", "linear"] = "relu",
        temperature: float = 1.0,
        threshold: float = 0.0,
        gene_hyperbolic: bool = False,
        eps: float = 1e-8,
        reduction: str | None = "mean",
    ) -> None:
        super().__init__()
        self.scaling_type = scaling_type
        self.temperature = temperature
        self.threshold = threshold
        self.gene_hyperbolic = gene_hyperbolic
        self.eps = eps
        self.reduction = reduction
        # A parameter-free helper for the optional hyperbolic affinity; None on the cosine path.
        self._hyp = WeightedContrastiveLoss(eps=eps) if gene_hyperbolic else None

    def _affinity(self, gene_embeddings: torch.Tensor) -> torch.Tensor:
        if self.gene_hyperbolic:
            sim = self._hyp._pairwise_poincare_similarity(gene_embeddings)
        else:
            sim = WeightedContrastiveLoss.cosine_distance(
                gene_embeddings, gene_embeddings
            )

        match self.scaling_type:
            case "relu":
                weight = sim.relu()
            case "linear":
                weight = (1.0 + sim) / 2.0
            case "sigmoid":
                weight = torch.sigmoid((sim - self.threshold) / self.temperature)
            case _:
                raise ValueError(f"Invalid scaling_type: {self.scaling_type!r}")

        # A node is not its own neighbour: zero the diagonal.
        return weight - torch.diag_embed(torch.diagonal(weight))

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        gene_embeddings: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # float32 so the Gram matrix and edge sums keep precision under bf16-mixed.
        y = y_pred.float()
        weight = self._affinity(gene_embeddings).float()

        gram = y @ y.T  # (B, B)
        sq = torch.diagonal(gram)  # (B,) ||y_i||^2
        deg = weight.sum(dim=1)  # (B,)

        # sum_ij W_ij ||y_i - y_j||^2 via the B x B Gram, never forming a G x G matrix.
        energy = 2.0 * (deg * sq).sum() - 2.0 * (weight * gram).sum()
        # The identity can dip slightly below zero from float rounding.
        energy = energy.clamp_min(0.0)

        total_w = weight.sum().clamp_min(self.eps)
        loss = energy / (total_w * y.shape[1])
        return loss.to(y_pred.dtype)
