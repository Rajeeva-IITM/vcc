"""Rectified-flow (conditional flow matching) model for the control -> perturbed route.

Instead of regressing the perturbed profile directly, this learns the *velocity* of a
straight-line path between a control cell and its paired perturbed cell:

    x_0 = control expression        x_1 = perturbed expression
    x_t = (1 - t) * x_0 + t * x_1           t ~ U[0, 1], one scalar per cell
    u   = x_1 - x_0                          (constant along a linear path)

and the net learns ``v(t, x_t, e) ~= E[u | x_t, e]`` for a perturbation embedding ``e``.
Prediction integrates that field from the control with a few Euler steps.

Two properties matter here. First, ``v -> 0`` reproduces the control *exactly*, so the
"predict no change" floor is reachable rather than structurally excluded -- the previous
gated-delta submission scored ``nmae = -2.43`` precisely because it emitted a small
constant delta it could not switch off. Second, because controls are paired to perturbed
cells at random, ``u`` is very noisy for fixed ``(x_t, e)``; the regression optimum is the
*marginal* velocity, which transports the control distribution onto the perturbed
distribution rather than mapping individual cells. That is what CFM targets, not a defect.
"""

import math
from pathlib import Path
from typing import Any, Literal

import polars.selectors as cs
import torch
from torch import nn

from src.models.components.basic_vcc_model import ProcessingNN
from src.utils.data import read_data
from src.utils.process_activation_function import get_activation


class SinusoidalTimeEmbedding(nn.Module):
    """Map a scalar time in [0, 1] to a vector of sines and cosines.

    Parameter-free. Each of the ``dim // 2`` frequency channels is a wave in ``t`` with a
    different wavelength, geometrically spaced, emitted as ``cat([sin, cos])``.

    Why bother, given ``t_processor`` is already an MLP: ``Linear(1, hidden)`` makes every
    hidden unit compute ``w_i * t + b_i``, so as ``t`` sweeps 0 -> 1 the hidden vector
    traces a *straight line* -- one degree of freedom about time no matter how wide the
    layer. Later nonlinearities can bend the output along that line, but gradient descent
    is slow at it (spectral bias). Sinusoidal features make ``t`` trace a curve visiting
    near-orthogonal directions, so a single linear readout separates "near t=0.2" from
    "near t=0.8".

    Parameters
    ----------
    dim : int
        Output width. Must be even; half the channels are sines, half cosines.
    max_period : float
        Ratio between the longest and shortest wavelength.
    scale : float
        **The one thing that must not be got wrong.** ``max_period=10000`` is the
        transformer *positional*-encoding convention, where the input is an integer
        position 0..10000. Here ``t`` lives in [0, 1], so with those frequencies the slow
        channels satisfy ``sin(1e-4 * t) ~= 0`` across the entire interval and most of the
        embedding is dead constants -- strictly worse than feeding the raw scalar. Scaling
        ``t`` up by 1000 first is the same thing diffusion codebases do by passing integer
        timesteps. The ``__main__`` block below asserts every channel actually varies.
    """

    def __init__(
        self, dim: int, max_period: float = 10000.0, scale: float = 1000.0
    ) -> None:
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"time embedding dim must be even, got {dim}")

        self.dim = dim
        self.scale = scale
        self.max_period = max_period

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        )
        # Not persistent: it is a pure function of the constructor args, so keeping it out
        # of the state dict means `dim`/`max_period` can be retuned without breaking
        # checkpoint loading.
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed a batch of times.

        Args:
            t (torch.Tensor): Times, shape ``(B,)`` or ``(B, 1)``, in [0, 1].

        Returns:
            torch.Tensor: Embedding of shape ``(B, dim)``.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        args = (t * self.scale) * self.freqs.to(dtype=t.dtype)

        return torch.cat([args.sin(), args.cos()], dim=-1)


class LearnableGeneEmbedding(nn.Module):
    """A per-gene embedding table that is trained rather than looked up frozen.

    The default perturbation representation is a fixed vector read off a parquet
    (Poincare/GO). This replaces the lookup with an ``nn.Embedding`` indexed by gene id,
    so gradient descent can reshape it. ``init_path`` decides where it starts.

    **Only genes that actually appear as perturbations receive gradient.** Everything else
    keeps whatever it was initialised to, and that is the whole reason ``init_path``
    matters: of the 300 targets in the 2026 panel, only 25 were perturbed in 2025. Started
    from the prior, the other 275 stay *exactly* at their prior value and the model
    degrades to today's behaviour on them. Started from scratch they stay random noise,
    which is a diagnostic run, not a submittable one.

    The risk this creates is worth naming. ``ko_processor`` only ever sees the ~240 rows
    that drifted, then at prediction time is handed un-drifted ones. The 2026 targets sit
    close to 2025 perturbations in prior space (nearest-neighbour cosine, median 0.958),
    so it is that geometry which carries the transfer -- and an unconstrained table is
    free to flatten it. :meth:`drift` exists to measure exactly that; ``embedding_lr`` in
    ``flow_lightning.VCCModule`` is the knob that limits it.

    Parameters
    ----------
    num_genes : int
        Rows in the table. Must match the initialisation table when one is given.
    embedding_dim : int
        Width of each embedding; feeds ``ko_processor_args["input_size"]``.
    init_path : str | Path | None
        Parquet with a ``gene_name`` column plus numeric embedding columns -- the same
        file the datamodule reads, so row *i* here is row *i* there. ``None`` gives the
        default random initialisation.
    freeze : bool
        Hold the table fixed. With ``init_path`` set this reproduces the frozen-lookup
        behaviour exactly, which is what the equivalence check in ``__main__`` asserts.
    """

    def __init__(
        self,
        num_genes: int,
        embedding_dim: int,
        init_path: str | Path | None = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()

        self.table = nn.Embedding(num_genes, embedding_dim)
        self.num_genes = num_genes
        self.embedding_dim = embedding_dim

        if init_path is not None:
            weights = torch.as_tensor(
                read_data(init_path).select(cs.numeric()).to_numpy(),
                dtype=self.table.weight.dtype,
            )
            if tuple(weights.shape) != (num_genes, embedding_dim):
                raise ValueError(
                    f"{init_path} holds a {tuple(weights.shape)} table but this module "
                    f"was built for ({num_genes}, {embedding_dim}); check `num_genes` "
                    f"and `embedding_dim` against the parquet"
                )
            with torch.no_grad():
                self.table.weight.copy_(weights)

            # Not persistent: it is a pure function of `init_path`, so keeping it out of
            # the state dict leaves the parquet as the single source of truth and stops a
            # 18 MB copy riding along in every checkpoint.
            self.register_buffer("prior", weights.clone(), persistent=False)

        self.table.weight.requires_grad = not freeze

    def forward(self, ko_id: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for a batch of gene ids.

        Args:
            ko_id (torch.Tensor): Gene ids, shape ``(B,)``, indexing the same row order as
                ``init_path``.

        Returns:
            torch.Tensor: Embeddings, shape ``(B, embedding_dim)``.
        """
        return self.table(ko_id)

    @torch.no_grad()
    def drift(self) -> tuple[float, int] | None:
        """How far the table has moved from its initialisation.

        Rows that never appear in a batch get no gradient and so do not move at all, which
        makes the set of moved rows self-selecting: the mean is taken over exactly the
        genes that are being trained. ``n_moved`` should equal the number of training
        perturbations, and is a free check that nothing else is quietly learning.

        Returns:
            tuple[float, int] | None: ``(mean relative L2 displacement over rows with a
            non-degenerate prior, rows moved)``, or
            ``None`` when the module was built without ``init_path`` and there is no
            reference to compare against.
        """
        prior = getattr(self, "prior", None)
        if prior is None:
            return None

        delta = (self.table.weight - prior).norm(dim=1)
        moved = delta > 0
        n_moved = int(moved.sum())
        if n_moved == 0:
            return 0.0, 0

        # 719 of the 18,080 rows in the GO parquet are exactly zero -- genes with no
        # annotation. Dividing by their norm gives ~1e7 and swamps the average: measured
        # on a real run, 86 such rows dragged a healthy 0.023 up to 65,520. Report the
        # relative move over rows that have a prior to be relative *to*; the zero rows
        # still count in `n_moved`. None of the 300 2026 targets is among them, so this
        # is a reporting fix and not a claim about the leaderboard; 86 of the 9,233
        # perturbed genes are.
        scale = prior[moved].norm(dim=1)
        usable = scale > 1e-6
        if not usable.any():
            return 0.0, n_moved

        return float((delta[moved][usable] / scale[usable]).mean()), n_moved


class KernelFiLM(nn.Module):
    """Kernel-smoothed conditioner: perturbation embedding -> FiLM params ``[gamma; beta]``.

    Drop-in for the ``ko_processor`` MLP in :class:`FlowCellModel`: same ``(B, in) ->
    (B, out)`` contract and the same ``.output_size`` / ``.input_size`` attributes, so it
    installs wherever the MLP did. What changes is the *shape* of the embedding -> response
    map. The MLP is free to carve a sharp basin per training gene, which is precisely the
    memorisation behind the two failures measured on the val panel: on held-out genes the
    direction collapses to the blind baseline (``fid``, ``pds``, ``jac`` ~ 0) **and** the
    magnitude blows up (``nmae`` at the floor even after centering, because the overshoot is
    perturbation-specific, not a common shift). This replaces the MLP with Nadaraya-Watson
    regression over ``num_anchors`` learned anchors::

        w(e)   = softmax_k( cos(e, key_k) / tau )      # (B, K), each row sums to 1
        out(e) = w(e) @ values                         # (B, out)

    Two structural consequences, one per failure:

    * **Smooth in the embedding.** ``out`` is a softmax-weighted average, hence a smooth
      function of ``e``: a held-out gene whose embedding sits near training genes gets a
      response *interpolated* from theirs rather than an MLP's arbitrary value. This is the
      "smooth embedding -> response map" the collapse diagnosis has always pointed to.
    * **Bounded magnitude.** ``out`` is a convex combination of the ``values`` rows, so it
      lives in their convex hull -- every output coordinate stays between the smallest and
      largest anchor value on that coordinate. It *structurally cannot* emit the 10x
      overshoot an MLP produces off-distribution, the part no post-hoc centering could fix.

    ``tau`` (kept as a learnable ``log_tau``) is the collapse<->memorise dial: large ``tau``
    -> near-uniform weights -> one shared response (the blind baseline); small ``tau`` ->
    near nearest-anchor lookup (memorisation). Left learnable so training finds the balance,
    but clamped at ``tau_min`` on read so a run cannot drive it to a degenerate spike.

    ``values`` is **zero-initialised**, so at init ``out = 0`` and the FiLM fusion
    ``(1 + gamma) * exp + beta`` is the identity on the state -- the same
    start-from-passthrough contract the MLP FiLM head has (its untrained head emits
    ``gamma, beta ~ 0``), so the perturbation is learned as a deviation from "no change".

    A count on parameters, since it flips the usual intuition: 256 anchors at
    ``input=256, out=512`` is ~0.2 M params, *fewer* than the 3-layer MLP it replaces (~1 M).
    The interpolation bias is a constraint, not extra capacity.

    Parameters
    ----------
    input_size : int
        Width of the perturbation embedding fed in (the ``gene_embedding`` output; 256 for
        the Poincare/GO lookup, matching ``ko_processor_args.input_size`` in the FiLM config).
    output_size : int
        Width emitted; for FiLM fusion this is ``2H`` = ``[gamma; beta]``.
    num_anchors : int
        Number of learned ``(key, value)`` anchors ``K``. A *modelling* choice, not the
        training-gene count: anchors are prototypes placed freely in embedding space, so
        held-out and training genes are scored against the same ``K`` responses.
    temperature : float
        Initial softmax bandwidth over cosine similarity (which lives in [-1, 1]).
    tau_min : float
        Lower clamp on ``tau`` at read time, so the softmax cannot sharpen into a hard argmax.
    learn_temperature : bool
        Whether ``tau`` receives gradient. ``False`` pins it at ``temperature``.
    proj_dim : int | None
        Rank of a LEARNED similarity metric. ``None`` (default) uses the fixed cosine above --
        the exact original module, and the state_dict is unchanged so old checkpoints load. An
        int ``d`` inserts a learned ``query_proj: (input_size -> d)`` so the weight becomes
        ``softmax(cos(W_Q e, key_k) / tau)`` with keys living in the ``d``-space -- a low-rank
        Mahalanobis similarity the model shapes to the response. This is the only change:
        ``out = weights @ values`` is untouched, so both structural properties above (bounded
        magnitude, smooth in ``e``) hold for any ``proj_dim``. Setting ``proj_dim = input_size``
        with ``W_Q = I`` would recover the identity metric, so the fixed-cosine head is its
        special case. A small ``d`` (e.g. 64) is a low-rank bottleneck -- a regulariser toward
        the few response-relevant directions, not extra capacity (it has *fewer* params than the
        identity metric, since the keys shrink from ``input_size`` to ``d``).
    extrapolate : bool
        If ``True`` (requires ``proj_dim``), replace the Nadaraya-Watson readout
        ``out = weights @ values`` -- degree-0, stuck in the anchor convex hull -- with **ridge
        local-linear** regression (degree-1): fit a plane ``values ~ b0 + B1.(key - q)`` weighted
        by the same anchor weights, in the ``proj_dim`` space, and read it at the query. The
        output then follows the local response *gradient* and can point OUTSIDE the hull, which is
        what a held-out gene needs when its true direction is not a blend of the anchors (the
        remaining ``fid``/direction wall). A learnable ridge ``log_lambda`` on the slope block
        keeps it safe: large ``lambda`` pins the slope to zero and recovers exact NW (the bounded
        ``-0.021`` head); small ``lambda`` is full extrapolation -- so training dials
        magnitude-safety against reach. ``values`` is still zero-init, so the plane is zero at
        init and the identity-FiLM start is preserved. Default ``False`` adds no parameter, so old
        checkpoints load unchanged.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_anchors: int = 256,
        temperature: float = 0.1,
        tau_min: float = 1e-2,
        learn_temperature: bool = True,
        proj_dim: int | None = None,
        context_dim: int | None = None,
        extrapolate: bool = False,
    ) -> None:
        super().__init__()

        if num_anchors < 1:
            raise ValueError(f"num_anchors must be >= 1, got {num_anchors}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if proj_dim is not None and proj_dim < 1:
            raise ValueError(f"proj_dim must be >= 1 or None, got {proj_dim}")
        if context_dim is not None and context_dim < 1:
            raise ValueError(f"context_dim must be >= 1 or None, got {context_dim}")
        if extrapolate and proj_dim is None:
            # Local-linear needs a per-query (1+d)x(1+d) solve; with d = input_size the system
            # is both intractable and rank-deficient (num_anchors < 1 + input_size). Require the
            # low-rank `proj_dim` space, where d is small and the solve is well-posed.
            raise ValueError(
                "extrapolate=True requires proj_dim (the local-linear metric space)"
            )

        self.input_size = input_size
        self.output_size = output_size
        self.num_anchors = num_anchors
        self.tau_min = tau_min
        self.proj_dim = proj_dim
        self.context_dim = context_dim
        self.extrapolate = extrapolate

        # `proj_dim` turns the fixed-cosine similarity into a LEARNED one: the query is first
        # projected into a `proj_dim` space by `query_proj` and the keys live there too, so the
        # weight `cos(W_Q e, key_k)` is a low-rank Mahalanobis similarity the model can shape to
        # the response geometry. `None` keeps the identity metric (keys in `input_size` space,
        # no `query_proj`), which is the exact original module -- and, because it adds no
        # parameter, keeps the state_dict byte-identical so pre-`proj_dim` checkpoints still
        # load under `strict=True`.
        key_dim = input_size if proj_dim is None else proj_dim
        if proj_dim is not None:
            self.query_proj = nn.Linear(input_size, proj_dim, bias=False)

        # `context_dim` makes the map `(e, context) -> params`: a shared encoder embeds the
        # cell-context vector (pooled control profile) into the query's key-space and ADDS it,
        # so the same perturbation routes to different anchors in different contexts. The last
        # layer is zero-init, so context contributes zero shift at start -> identical to the
        # no-context module, and (like `proj_dim`) `None` adds no parameter, keeping the
        # state_dict byte-identical for strict-load. `out = weights @ values` is untouched, so
        # the convex-hull magnitude bound and smoothness both survive.
        if context_dim is not None:
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim, key_dim),
                nn.SiLU(),
                nn.Linear(key_dim, key_dim),
            )
            nn.init.zeros_(self.context_encoder[-1].weight)
            nn.init.zeros_(self.context_encoder[-1].bias)

        # Keys tile the (projected) embedding manifold; random init is fine because softmax
        # makes the map smooth in `e` regardless of where the keys sit, and they get gradient.
        self.keys = nn.Parameter(torch.randn(num_anchors, key_dim))
        # Zero values -> zero output at init -> identity FiLM (see class docstring).
        self.values = nn.Parameter(torch.zeros(num_anchors, output_size))

        log_tau = torch.log(torch.tensor(float(temperature)))
        if learn_temperature:
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_tau", log_tau)

        # Diagnostic stash: mean normalised entropy of the anchor weights on the last
        # forward, in [0, 1]. ~1 means weights are ~uniform (collapsing toward the blind
        # baseline); ~0 means each cell locks onto a single anchor (memorising). Read by
        # flow_lightning and logged as `*/kernel_entropy` -- the direct readout of the
        # collapse<->memorise axis this whole change is about. Detached; not persistent.
        self.register_buffer("last_entropy", torch.zeros(()), persistent=False)

        # `extrapolate` upgrades the readout from Nadaraya-Watson (degree-0, `weights @ values`,
        # output stuck in the anchor convex hull) to ridge local-linear (degree-1): fit a local
        # plane through the weighted anchors and read it at the query, so the output follows the
        # local response GRADIENT and can leave the hull -- the boundary-bias correction that lets
        # held-out genes point in directions no anchor holds. `log_lambda` is a learnable ridge on
        # the slope block: large lambda -> slope forced to 0 -> exact NW (bounded, the -0.021
        # head); small lambda -> full extrapolation. So training dials magnitude-safety <-> reach,
        # the magnitude analogue of tau's collapse<->memorise. Init lambda=1 starts near NW; with
        # zero-init `values` the plane is 0 at init either way, so identity-FiLM is preserved.
        if extrapolate:
            self.log_lambda = nn.Parameter(torch.zeros(()))

    def forward(
        self, e: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Map embeddings (and optional context) to FiLM params via kernel regression.

        Args:
            e (torch.Tensor): Perturbation embeddings, shape ``(B, input_size)``. A zero row
                (an unannotated gene in the GO parquet) normalises to zero, giving uniform
                weights and hence the mean anchor response -- a sensible blind fallback.
            context (torch.Tensor | None): Cell-context vector, shape ``(B, context_dim)``.
                Required iff the module was built with ``context_dim``; ignored otherwise.

        Returns:
            torch.Tensor: FiLM params, shape ``(B, output_size)``.
        """
        # Learned metric: project the query into the keys' space first. A zero input row still
        # projects to zero -> uniform weights -> the mean anchor response, the same blind
        # fallback as the identity metric.
        q = self.query_proj(e) if self.proj_dim is not None else e
        # Context shifts the query -> different anchor routing per context. Zero-init encoder
        # means no shift at start.
        if self.context_dim is not None:
            if context is None:
                raise ValueError("KernelFiLM built with context_dim needs a `context`")
            q = q + self.context_encoder(context)
        q = nn.functional.normalize(q, dim=-1)
        keys = nn.functional.normalize(self.keys, dim=-1)
        tau = self.log_tau.exp().clamp_min(self.tau_min)

        weights = torch.softmax((q @ keys.t()) / tau, dim=-1)  # (B, K)

        with torch.no_grad():
            p = weights.clamp_min(1e-12)
            entropy = -(p * p.log()).sum(-1) / math.log(self.num_anchors)
            self.last_entropy = entropy.mean()

        if not self.extrapolate:
            return (
                weights @ self.values
            )  # (B, output_size) -- Nadaraya-Watson (in the hull)

        # Ridge local-linear regression: fit `values ~ b0 + B1 . (key - q)` weighted by `weights`
        # in the projected key-space, then read the plane at the query (regressors centred at q, so
        # the prediction is the intercept b0). This is degree-1 kernel regression -- it extrapolates
        # along the local response gradient instead of averaging inside the anchor hull.
        # The solve runs in fp32 with autocast OFF: `linalg.solve` is autocast-excluded and
        # dtype-strict, and the weighted-Gram solve is numerically sensitive under bf16 anyway.
        out_dtype = weights.dtype
        with torch.autocast(device_type=q.device.type, enabled=False):
            q32, keys32 = q.float(), keys.float()
            w32, v32 = weights.float(), self.values.float()
            d = keys32.unsqueeze(0) - q32.unsqueeze(
                1
            )  # (B, K, key_dim): anchor offsets from query
            X = torch.cat([torch.ones_like(d[..., :1]), d], dim=-1)  # (B, K, 1+key_dim)
            XtW = X * w32.unsqueeze(-1)  # weight each anchor row
            XtWX = XtW.transpose(1, 2) @ X  # (B, 1+key_dim, 1+key_dim)
            XtWy = XtW.transpose(1, 2) @ v32  # (B, 1+key_dim, output_size)
            ridge = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)
            ridge[0, 0] = 0.0  # never penalise the intercept (the prediction itself)
            XtWX = XtWX + self.log_lambda.float().exp() * ridge
            beta = torch.linalg.solve(XtWX, XtWy)  # (B, 1+key_dim, output_size)
        return beta[:, 0, :].to(
            out_dtype
        )  # (B, output_size): the local plane at the query


class FiLMProcessingNN(nn.Module):
    """A conditioned trunk that FiLM-modulates *every* hidden layer, not just once.

    The single-FiLM :class:`FlowCellModel` (``fusion_type='film'``) conditions once -- one
    ``(gamma, beta)`` on the state latent, then a plain decoder. This is how conditioning is
    actually done in generative models (diffusion U-Nets, StyleGAN AdaIN): the perturbation
    reshapes the representation at *each* block. Given a state ``x`` and a conditioning vector
    ``c`` (the perturbation embedding), with ``L`` layers of width ``H``::

        h_0 = W_"in" x  ( + t_emb )
        h_l = act( (1 + gamma_l) dot.circle LN(W_l h_(l-1)) + beta_l )  [ + h_(l-1) ]

    All ``L`` pairs ``(gamma_l, beta_l)`` come from one call to ``conditioner(c)``, which must
    emit width ``2 H L``; they are sliced per layer. Passing a :class:`KernelFiLM` as the
    conditioner is the point: its output is a convex combination of learned anchor values, so
    **every layer's modulation is bounded** (the anti-overshoot property that fixed ``nmae``,
    now enforced at depth) and **smooth in the embedding** (held-out genes interpolate). A
    per-layer *MLP* conditioner would be unbounded at every layer and reintroduce the 10x
    overshoot, compounded ``L`` times -- so prefer the kernel.

    Identity-of-conditioning at init: a zero-value KernelFiLM emits ``gamma_l = beta_l = 0``,
    so the perturbation contributes nothing and the trunk computes an unconditioned transform
    of ``x``; the perturbation is then learned as a deviation. (The trunk itself is not the
    identity -- it is a decoder -- which is expected and correct.)

    Parameters
    ----------
    input_size : int
        Width of the state ``x`` fed in (``n_genes`` for the flow).
    hidden_size : int
        Trunk width ``H``. The time embedding, when supplied, is added at this width.
    num_hidden_layers : int
        Number of FiLM-conditioned layers ``L``.
    output_size : int
        Width emitted (``n_genes`` -- the velocity). The output head is a plain ``Linear``:
        the velocity is sign-free, so no positivity enforcer is added.
    conditioner : nn.Module
        Maps the conditioning vector ``c`` to the stacked per-layer params; must expose
        ``.output_size == 2 * hidden_size * num_hidden_layers``.
    dropout : float
        Dropout after each layer.
    activation : str | nn.Module
        Per-layer activation.
    residual_connection : bool
        Add a skip ``h <- h + h_(l-1)`` around each conditioned layer (helps gradients flow
        through ``L`` blocks). On by default.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        output_size: int,
        conditioner: nn.Module,
        dropout: float = 0.0,
        activation: str | nn.Module = "gelu",
        residual_connection: bool = True,
    ) -> None:
        super().__init__()

        expected = 2 * hidden_size * num_hidden_layers
        if conditioner.output_size != expected:
            raise ValueError(
                f"conditioner must emit [gamma_l; beta_l] for every layer, i.e. "
                f"2 * hidden_size * num_hidden_layers = {expected}; got "
                f"conditioner.output_size = {conditioner.output_size}"
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.output_size = output_size
        self.residual_connection = residual_connection

        self.conditioner = conditioner
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_size) for _ in range(num_hidden_layers)
        )
        self.output_projection = nn.Linear(hidden_size, output_size)
        self.activation = (
            get_activation(activation) if isinstance(activation, str) else activation
        )
        self.dropout = nn.Dropout(dropout)

        # Mirror the conditioner's entropy stash so flow_lightning can log it via
        # `self.net.trunk.last_entropy` exactly as it does for a top-level KernelFiLM.
        self.register_buffer("last_entropy", torch.zeros(()), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t_emb: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the conditioned trunk.

        Args:
            x (torch.Tensor): State, shape ``(B, input_size)``.
            c (torch.Tensor): Conditioning vector (perturbation embedding), shape
                ``(B, cond_in)`` as accepted by ``conditioner``.
            t_emb (torch.Tensor | None): Optional time latent, shape ``(B, hidden_size)``,
                added at the trunk input.
            context (torch.Tensor | None): Optional cell-context vector forwarded to the
                conditioner, so every layer's FiLM params depend on (perturbation, context)
                jointly. Only used when the conditioner was built to accept it (a
                ``context_dim``-equipped :class:`KernelFiLM`); ``None`` reproduces the
                perturbation-only path exactly.

        Returns:
            torch.Tensor: Output, shape ``(B, output_size)``. Sign-unconstrained.
        """
        # Pass context only when given, so conditioners that take a single argument still work.
        params = (
            self.conditioner(c, context) if context is not None else self.conditioner(c)
        )
        params = params.view(x.shape[0], self.num_hidden_layers, 2, self.hidden_size)
        gammas = params[:, :, 0, :]  # (B, L, H)
        betas = params[:, :, 1, :]

        entropy = getattr(self.conditioner, "last_entropy", None)
        if entropy is not None:
            self.last_entropy = entropy.detach()

        h = self.input_projection(x)
        if t_emb is not None:
            h = h + t_emb

        for i in range(self.num_hidden_layers):
            residual = h
            h = self.norms[i](self.layers[i](h))
            h = (1.0 + gammas[:, i]) * h + betas[:, i]
            h = self.activation(h)
            if self.residual_connection:
                h = h + residual
            h = self.dropout(h)

        return self.output_projection(h)


class FlowCellModel(nn.Module):
    """Velocity field ``v(t, x_t, ko_vec)`` for the control -> perturbed flow.

    Three encoders (time, perturbation, current state) are fused and decoded to a
    full-width velocity over genes. The velocity is deliberately **unconstrained in sign**:
    a knockdown drives its target gene down, so a non-negative output could not express the
    single most important effect in the data.

    Parameters
    ----------
    t_processor_args : dict
        ``ProcessingNN`` args for the time branch. ``input_size`` must equal
        ``time_embedding_dim``.
    ko_processor_args : dict | None
        ``ProcessingNN`` args for the perturbation embedding branch (the MLP conditioner).
        Give this **or** ``ko_conditioner``, not both.
    ko_conditioner : nn.Module | None
        An injected perturbation branch replacing the MLP -- e.g. :class:`KernelFiLM`, which
        maps the embedding to ``[gamma; beta]`` by kernel regression over learned anchors
        instead of an MLP, so the map is smooth in the embedding and bounded in magnitude.
        Must expose ``.output_size``; under ``film`` fusion that must be ``2H``.
    exp_processor_args : dict
        ``ProcessingNN`` args for the current-state branch. Receives ``x_t``, *not* the
        control.
    fused_processor_args : dict
        ``ProcessingNN`` args mapping the fused latent to the per-gene velocity. Leave
        ``ensure_output_positive`` false.
    fusion_type : {"sum", "product", "concat", "film", "film_deep"}
        How the three branches combine. ``sum``/``product`` require all three branches to
        share one width and combine them elementwise; ``concat`` stacks them. ``film`` is
        different in kind: the perturbation stops being a branch that is *added* and becomes
        a FiLM operator that *modulates the state*. ``ko_processor`` then emits ``[gamma;
        beta]`` (width ``2H``) and the fusion is ``(1 + gamma) * exp_processed + beta``,
        with time added afterward. The ``1 +`` makes it identity at init (gamma, beta ~ 0
        from the untrained linear head), so the state passes through untouched and the
        perturbation is learned as a deviation from it. This directly targets perturbation
        specificity: an additive perturbation signal averages away over cells and the model
        collapses to the mean profile, whereas a multiplicative operator cannot. ``film_deep``
        goes further -- a single :class:`FiLMProcessingNN` ``trunk`` FiLM-modulates *every*
        layer (generative-model style) instead of once. It uses ``trunk`` and the time branch
        only; ``exp_processor_args``, ``fused_processor_args`` and the ``ko`` args are unused.
    trunk : nn.Module | None
        Required iff ``fusion_type='film_deep'``: a :class:`FiLMProcessingNN` that consumes
        the state and the perturbation and emits the velocity. Its ``hidden_size`` must equal
        ``t_processor.output_size`` (the time latent is added at the trunk input).
    gene_embedding : nn.Module | None
        Applied to the perturbation input before ``ko_processor``. ``None`` means
        ``nn.Identity``, i.e. the input already *is* the embedding vector -- the frozen
        lookup this model has always used. Pass a
        :class:`LearnableGeneEmbedding` to instead feed gene *ids* and train the table.
        The two cases share one code path; only what the caller puts in ``ko`` changes.
    time_embedding_dim : int
        Width of the sinusoidal time embedding.
    time_embedding_scale : float
        See :class:`SinusoidalTimeEmbedding`.
    """

    def __init__(
        self,
        t_processor_args: dict[str, Any],
        fusion_type: Literal["sum", "product", "concat", "film", "film_deep"],
        exp_processor_args: dict[str, Any] | None = None,
        fused_processor_args: dict[str, Any] | None = None,
        ko_processor_args: dict[str, Any] | None = None,
        ko_conditioner: nn.Module | None = None,
        trunk: nn.Module | None = None,
        gene_embedding: nn.Module | None = None,
        time_embedding_dim: int = 128,
        time_embedding_scale: float = 1000.0,
    ) -> None:
        super().__init__()

        self.gene_embedding = (
            nn.Identity() if gene_embedding is None else gene_embedding
        )
        self.time_embedding = SinusoidalTimeEmbedding(
            dim=time_embedding_dim, scale=time_embedding_scale
        )
        self.fusion = fusion_type

        # `film_deep` is a different animal from the other fusions: instead of three branches
        # combined once, a single conditioned trunk (:class:`FiLMProcessingNN`) FiLM-modulates
        # every layer from the perturbation. It owns its conditioner, so `ko_processor_args`,
        # `ko_conditioner`, `exp_processor_args` and `fused_processor_args` are all unused; the
        # time branch is still built and added at the trunk input.
        if fusion_type == "film_deep":
            if trunk is None:
                raise ValueError(
                    "fusion_type='film_deep' requires a `trunk` (a FiLMProcessingNN)"
                )
            if t_processor_args["input_size"] != time_embedding_dim:
                raise ValueError(
                    f"t_processor input_size ({t_processor_args['input_size']}) must equal "
                    f"time_embedding_dim ({time_embedding_dim})"
                )
            if t_processor_args["output_size"] != trunk.hidden_size:
                raise ValueError(
                    f"fusion_type='film_deep' adds the time branch to the trunk input, so "
                    f"t_processor.output_size must equal trunk.hidden_size; got "
                    f"t={t_processor_args['output_size']}, trunk={trunk.hidden_size}"
                )
            self.t_processor = ProcessingNN(**t_processor_args)
            self.trunk = trunk
            return

        # The perturbation branch is either an MLP built from `ko_processor_args` (the
        # historical path) or an injected module such as `KernelFiLM` passed as
        # `ko_conditioner`. Exactly one; both interfaces are `(B, in) -> (B, out)` with an
        # `.output_size`, so everything downstream is identical.
        if exp_processor_args is None or fused_processor_args is None:
            raise ValueError(
                f"fusion_type={fusion_type!r} needs both exp_processor_args and "
                f"fused_processor_args (only 'film_deep' omits them)"
            )
        if (ko_processor_args is None) == (ko_conditioner is None):
            raise ValueError(
                "provide exactly one of `ko_processor_args` (build an MLP) or "
                "`ko_conditioner` (inject a module, e.g. KernelFiLM)"
            )

        if t_processor_args["input_size"] != time_embedding_dim:
            raise ValueError(
                f"t_processor input_size ({t_processor_args['input_size']}) must equal "
                f"time_embedding_dim ({time_embedding_dim}); the time branch is fed the "
                f"sinusoidal embedding, not the raw scalar"
            )

        # CellModel leaves the equivalent check commented out and it is the error people
        # actually hit, so make it loud and specific here.
        ko_output_size = (
            ko_conditioner.output_size
            if ko_conditioner is not None
            else ko_processor_args["output_size"]
        )
        branch_sizes = (
            t_processor_args["output_size"],
            ko_output_size,
            exp_processor_args["output_size"],
        )
        match fusion_type:
            case "concat":
                expected = sum(branch_sizes)
            case "sum" | "product":
                if len(set(branch_sizes)) != 1:
                    raise ValueError(
                        f"fusion_type={fusion_type!r} adds the three branches elementwise, "
                        f"so their output_sizes must all match; got t={branch_sizes[0]}, "
                        f"ko={branch_sizes[1]}, exp={branch_sizes[2]}"
                    )
                expected = branch_sizes[0]
            case "film":
                # The perturbation is a FiLM operator on the state rather than a term added
                # to it: h <- (1 + gamma(p)) * exp_processed + beta(p), then time is added.
                # So the state branch sets the latent width H; ko must emit [gamma; beta]
                # (2H); time is added to h (H); and the decoder consumes h (H).
                t_out, ko_out, exp_out = branch_sizes
                if ko_out != 2 * exp_out:
                    raise ValueError(
                        f"fusion_type='film' feeds ko_processor's output as [gamma; beta], "
                        f"so its output_size must be twice exp_processor's; got "
                        f"ko={ko_out}, exp={exp_out} (expected ko={2 * exp_out})"
                    )
                if t_out != exp_out:
                    raise ValueError(
                        f"fusion_type='film' adds the time branch to the modulated state, "
                        f"so t_processor.output_size must equal exp_processor's; got "
                        f"t={t_out}, exp={exp_out}"
                    )
                expected = exp_out
            case _:
                raise ValueError(
                    "fusion_type should be one of ['sum', 'product', 'concat', 'film'], "
                    f"got {fusion_type!r}"
                )

        if fused_processor_args["input_size"] != expected:
            raise ValueError(
                f"fused_processor input_size must be {expected} for fusion_type="
                f"{fusion_type!r}, got {fused_processor_args['input_size']}"
            )

        # `gene_embedding` (Identity by default) keeps the historical contract: `ko` arrives
        # as the precomputed vector. It holds no parameters, so checkpoints from before this
        # argument existed still load unchanged.
        self.t_processor = ProcessingNN(**t_processor_args)
        self.ko_processor = (
            ko_conditioner
            if ko_conditioner is not None
            else ProcessingNN(**ko_processor_args)
        )
        self.exp_processor = ProcessingNN(**exp_processor_args)
        self.fused_processor = ProcessingNN(**fused_processor_args)
        # Only a conditioner (KernelFiLM) accepts a `context` arg; an MLP ko_processor does not.
        self._ko_conditioned = ko_conditioner is not None

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        ko: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the velocity field.

        Args:
            t (torch.Tensor): Times, shape ``(B, 1)``, in [0, 1].
            x_t (torch.Tensor): Current state on the path, shape ``(B, n_genes)``. This is
                the interpolant during training and the running sample during integration
                -- never the control, except at ``t = 0``.
            ko (torch.Tensor): The perturbation, in whatever form ``gene_embedding``
                consumes: embedding vectors ``(B, embed_dim)`` by default, or gene ids
                ``(B,)`` when a :class:`LearnableGeneEmbedding` is installed.

        Returns:
            torch.Tensor: Velocity, shape ``(B, n_genes)``. Sign-unconstrained.
        """
        if self.fusion == "film_deep":
            # One conditioned trunk: the perturbation FiLM-modulates every layer, the time
            # latent is added at the trunk input. No separate ko/exp branches to fuse.
            # `context` is forwarded so a context_dim-equipped conditioner makes each layer's
            # FiLM params depend on (perturbation, context); None keeps the old behaviour.
            t_processed = self.t_processor(self.time_embedding(t))
            return self.trunk(x_t, self.gene_embedding(ko), t_processed, context)

        t_processed = self.t_processor(self.time_embedding(t))
        ko_processed = (
            self.ko_processor(self.gene_embedding(ko), context)
            if self._ko_conditioned
            else self.ko_processor(self.gene_embedding(ko))
        )
        exp_processed = self.exp_processor(x_t)

        match self.fusion:
            case "sum":
                fused_representation = t_processed + ko_processed + exp_processed

            case "product":
                fused_representation = t_processed * ko_processed * exp_processed

            case "concat":
                fused_representation = torch.cat(
                    [t_processed, ko_processed, exp_processed], dim=1
                )

            case "film":
                # ko_processed is [gamma; beta]; each half has the width of exp_processed.
                # (1 + gamma) so an untrained head (gamma, beta ~ 0) leaves the state
                # untouched -- the perturbation is a learned deviation, not a from-scratch
                # gate. Time is added after modulation: it is positional, not semantic, so
                # it should shift the field rather than scale the state.
                gamma, beta = ko_processed.chunk(2, dim=-1)
                fused_representation = (
                    (1.0 + gamma) * exp_processed + beta + t_processed
                )

            case _:
                raise ValueError(
                    "fusion_type should be one of ['sum', 'product', 'concat', 'film']"
                )

        return self.fused_processor(fused_representation)

    def step(
        self,
        x: torch.Tensor,
        ko: torch.Tensor,
        t_start: float,
        t_end: float,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Take one explicit Euler step of the ODE ``dx/dt = v(t, x, ko)``.

        Args:
            x (torch.Tensor): Current state, shape ``(B, n_genes)``.
            ko (torch.Tensor): Perturbation, as accepted by :meth:`forward`.
            t_start (float): Time at the start of the step.
            t_end (float): Time at the end of the step.
            context (torch.Tensor | None): Cell-context vector, passed to :meth:`forward`.

        Returns:
            torch.Tensor: State at ``t_end``.
        """
        t = torch.full((x.shape[0], 1), float(t_start), device=x.device, dtype=x.dtype)

        return x + (t_end - t_start) * self.forward(t, x, ko, context)

    @torch.no_grad()
    def sample(
        self,
        x0: torch.Tensor,
        ko: torch.Tensor,
        num_steps: int = 4,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate the field from the control to ``t = 1`` to get a perturbed profile.

        Args:
            x0 (torch.Tensor): Control expression, shape ``(B, n_genes)``.
            ko (torch.Tensor): Perturbation, as accepted by :meth:`forward`.
            num_steps (int): Number of uniform Euler steps.

        Returns:
            torch.Tensor: Predicted perturbed expression, clamped to be non-negative --
            the inputs are ``log1p`` of a non-negative quantity, so the output must be
            too. This is the only place positivity is enforced; the velocity itself must
            stay free to be negative.
        """
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        x = x0
        dt = 1.0 / num_steps
        for i in range(num_steps):
            x = self.step(x, ko, i * dt, (i + 1) * dt, context)

        return x.clamp_min(0.0)


if __name__ == "__main__":
    # Smoke test: shapes, and the four properties that make this a flow rather than the
    # conditional regressor it was before.
    torch.manual_seed(0)

    n_genes, embed_dim, batch, t_dim = 512, 256, 8, 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _args(inp: int, out: int, hidden: int = 64, layers: int = 2) -> dict[str, Any]:
        """Build a small ProcessingNN arg dict for the smoke test."""
        return {
            "input_size": inp,
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "output_size": out,
            "dropout": 0.0,
            "activation": "gelu",
        }

    model = FlowCellModel(
        t_processor_args=_args(t_dim, 128),
        ko_processor_args=_args(embed_dim, 128),
        exp_processor_args=_args(n_genes, 128),
        fused_processor_args=_args(128, n_genes, hidden=256),
        fusion_type="sum",
        time_embedding_dim=t_dim,
    ).to(device)
    model.eval()

    x_t = torch.rand(batch, n_genes, device=device)
    ko = torch.randn(batch, embed_dim, device=device)
    t = torch.rand(batch, 1, device=device)

    # 1. shapes
    v = model(t, x_t, ko)
    assert v.shape == (batch, n_genes), v.shape
    assert torch.isfinite(v).all()

    # 2. the bug that mattered: is `t` actually used? Under the old code the net never saw
    #    x_t, so t was independent of the input and the optimum ignored it entirely.
    v_lo = model(torch.full((batch, 1), 0.1, device=device), x_t, ko)
    v_hi = model(torch.full((batch, 1), 0.9, device=device), x_t, ko)
    assert not torch.allclose(v_lo, v_hi), "velocity does not depend on t"

    # 3. sign freedom -- a non-negative velocity cannot express a knocked-down gene.
    assert (v < 0).any(), "velocity never goes negative; is ensure_output_positive set?"

    # 4. sampling
    out = model.sample(x_t, ko, num_steps=4)
    assert out.shape == (batch, n_genes) and torch.isfinite(out).all()
    assert (out >= 0).all()

    # 5. zero-velocity floor: this is the property that rescues `nmae`.
    with torch.no_grad():
        model.fused_processor.output_projection.weight.zero_()
        model.fused_processor.output_projection.bias.zero_()
    assert torch.allclose(model.sample(x_t, ko, num_steps=4), x_t, atol=1e-6), (
        "zero velocity must reproduce the control exactly"
    )

    # 6. the time embedding is alive -- catches the max_period scaling trap, where the slow
    #    channels become dead constants over t in [0, 1].
    emb = SinusoidalTimeEmbedding(dim=t_dim).to(device)
    grid = torch.linspace(0, 1, 32, device=device).unsqueeze(-1)
    channel_std = emb(grid).std(dim=0)
    assert channel_std.min() > 1e-3, (
        f"dead time-embedding channels (min std {channel_std.min():.2e}); "
        f"check SinusoidalTimeEmbedding.scale against max_period"
    )
    assert not torch.allclose(emb(grid[:1]), emb(grid[-1:]))

    # 7. the learnable table: ids in, embeddings out, and it integrates.
    n_table = 64
    learn = LearnableGeneEmbedding(num_genes=n_table, embedding_dim=embed_dim)
    ko_id = torch.randint(0, n_table, (batch,), device=device)
    model_learn = FlowCellModel(
        t_processor_args=_args(t_dim, 128),
        ko_processor_args=_args(embed_dim, 128),
        exp_processor_args=_args(n_genes, 128),
        fused_processor_args=_args(128, n_genes, hidden=256),
        fusion_type="sum",
        gene_embedding=learn,
        time_embedding_dim=t_dim,
    ).to(device)
    model_learn.eval()

    v_learn = model_learn(t, x_t, ko_id)
    assert v_learn.shape == (batch, n_genes) and torch.isfinite(v_learn).all()
    assert model_learn.sample(x_t, ko_id, num_steps=4).shape == (batch, n_genes)

    # 8. equivalence -- this is the property the whole design rests on. A frozen table fed
    #    ids must give bit-identical output to the same net fed the vectors directly, so
    #    `gene_embedding` genuinely only changes *how the perturbation is addressed*.
    plain = FlowCellModel(
        t_processor_args=_args(t_dim, 128),
        ko_processor_args=_args(embed_dim, 128),
        exp_processor_args=_args(n_genes, 128),
        fused_processor_args=_args(128, n_genes, hidden=256),
        fusion_type="sum",
        time_embedding_dim=t_dim,
    ).to(device)
    plain.eval()
    plain.load_state_dict(
        {k: v for k, v in model_learn.state_dict().items() if "gene_embedding" not in k}
    )
    assert torch.equal(plain(t, x_t, learn.table.weight[ko_id]), v_learn), (
        "id lookup and direct vectors must agree"
    )

    # 9. an Identity-embedding checkpoint round-trips into a model built without the
    #    argument at all, so runs from before this existed still load.
    plain.load_state_dict(model.state_dict())

    # 10. drift(): zero at init, and it counts only the rows that actually moved.
    with torch.no_grad():
        table = LearnableGeneEmbedding(num_genes=4, embedding_dim=8, init_path=None)
        table.register_buffer("prior", table.table.weight.clone(), persistent=False)
        assert table.drift() == (0.0, 0)
        table.table.weight[2] += 1.0
        drift, n_moved = table.drift()
        assert n_moved == 1 and drift > 0, (drift, n_moved)
    assert LearnableGeneEmbedding(4, 8).drift() is None, (
        "no init_path means no reference to measure drift against"
    )

    # 11. FiLM fusion: the perturbation modulates the state instead of being added to it.
    #     H is the state-latent width; ko emits [gamma; beta] (2H), time is added (H).
    H = 128
    film = FlowCellModel(
        t_processor_args=_args(t_dim, H),
        ko_processor_args=_args(embed_dim, 2 * H),
        exp_processor_args=_args(n_genes, H),
        fused_processor_args=_args(H, n_genes, hidden=256),
        fusion_type="film",
        time_embedding_dim=t_dim,
    ).to(device)
    film.eval()

    v_film = film(t, x_t, ko)
    assert v_film.shape == (batch, n_genes) and torch.isfinite(v_film).all()
    # velocity actually depends on t, and can go negative (knockdowns), and integrates.
    assert not torch.allclose(
        film(torch.full((batch, 1), 0.1, device=device), x_t, ko),
        film(torch.full((batch, 1), 0.9, device=device), x_t, ko),
    ), "FiLM velocity does not depend on t"
    assert (v_film < 0).any(), "FiLM velocity never negative"
    out_film = film.sample(x_t, ko, num_steps=4)
    assert out_film.shape == (batch, n_genes) and (out_film >= 0).all()

    # identity-init: with gamma, beta and the time branch zeroed, the (1 + gamma) form must
    # pass the state through untouched, so forward reduces to fused_processor(exp(x_t)).
    with torch.no_grad():
        film.ko_processor.output_projection.weight.zero_()
        film.ko_processor.output_projection.bias.zero_()
        film.t_processor.output_projection.weight.zero_()
        film.t_processor.output_projection.bias.zero_()
        expected_film = film.fused_processor(film.exp_processor(x_t))
    assert torch.allclose(film(t, x_t, ko), expected_film, atol=1e-6), (
        "(1 + gamma) FiLM must be identity on the state when gamma=beta=0"
    )

    # the film branch validates its own dimensions: ko must be exactly 2 * exp.
    try:
        FlowCellModel(
            t_processor_args=_args(t_dim, H),
            ko_processor_args=_args(embed_dim, 2 * H + 1),  # deliberately wrong
            exp_processor_args=_args(n_genes, H),
            fused_processor_args=_args(H, n_genes, hidden=256),
            fusion_type="film",
            time_embedding_dim=t_dim,
        )
        raise AssertionError("film fusion accepted ko_output != 2 * exp_output")
    except ValueError:
        pass

    # 12. KernelFiLM: the interpolating conditioner. Same (B, in) -> (B, out) contract as
    #     the ko MLP, but with the two structural properties the val panel says we need.
    kf = KernelFiLM(input_size=embed_dim, output_size=2 * H, num_anchors=32).to(device)
    kf.eval()
    kf_out = kf(ko)
    assert kf_out.shape == (batch, 2 * H) and torch.isfinite(kf_out).all()

    # zero-value init -> zero output -> identity FiLM downstream.
    assert torch.count_nonzero(kf_out) == 0, "KernelFiLM must emit zeros at init"

    # bounded magnitude: output is a convex combination of `values`, so every coordinate
    # lands in [min_k values, max_k values]. This is the anti-overshoot property.
    with torch.no_grad():
        kf.values.copy_(torch.randn_like(kf.values))
    kf_out = kf(ko)
    lo = kf.values.min(dim=0).values
    hi = kf.values.max(dim=0).values
    assert (kf_out >= lo - 1e-5).all() and (kf_out <= hi + 1e-5).all(), (
        "KernelFiLM output escaped the convex hull of its anchor values"
    )

    # entropy diagnostic is populated and normalised into [0, 1].
    assert 0.0 <= float(kf.last_entropy) <= 1.0 + 1e-5, kf.last_entropy

    # smooth in the embedding: a small input nudge gives a small output change (softmax is
    # continuous), unlike an MLP that can jump across a memorised basin boundary.
    with torch.no_grad():
        d = kf(ko + 1e-3 * torch.randn_like(ko)) - kf(ko)
    assert d.abs().max() < 0.5, (
        f"KernelFiLM not smooth: max |delta| {d.abs().max():.3f}"
    )

    # backward-compat: the default (identity-metric) module must add NO `query_proj` param and
    # keep `keys` at (K, input_size), so its state_dict is byte-identical to pre-`proj_dim`
    # checkpoints and still loads under strict=True.
    default_keys = set(KernelFiLM(embed_dim, 2 * H, num_anchors=32).state_dict())
    assert not any(k.startswith("query_proj") for k in default_keys), default_keys
    assert kf.keys.shape == (32, embed_dim), kf.keys.shape

    # 12b. Learned metric (`proj_dim`): the same structural properties must survive the added
    #      projection. Keys now live in the proj_dim space and a `query_proj` weight appears.
    kfm = KernelFiLM(
        input_size=embed_dim, output_size=2 * H, num_anchors=32, proj_dim=8
    ).to(device)
    kfm.eval()
    kfm_out = kfm(ko)
    assert kfm_out.shape == (batch, 2 * H) and torch.isfinite(kfm_out).all()
    assert torch.count_nonzero(kfm_out) == 0, (
        "proj_dim KernelFiLM must emit zeros at init"
    )
    sd = kfm.state_dict()
    assert "query_proj.weight" in sd and tuple(sd["query_proj.weight"].shape) == (
        8,
        embed_dim,
    )
    assert kfm.keys.shape == (32, 8), kfm.keys.shape
    with torch.no_grad():
        kfm.values.copy_(torch.randn_like(kfm.values))
    kfm_out = kfm(ko)
    lo, hi = kfm.values.min(dim=0).values, kfm.values.max(dim=0).values
    assert (kfm_out >= lo - 1e-5).all() and (kfm_out <= hi + 1e-5).all(), (
        "proj_dim KernelFiLM output escaped the convex hull of its anchor values"
    )
    assert 0.0 <= float(kfm.last_entropy) <= 1.0 + 1e-5, kfm.last_entropy

    # 12c. Extrapolation (`extrapolate`): the DEFINING property is that the output can leave the
    #      anchor convex hull (degree-1 local-linear), and that a large ridge recovers NW (in the
    #      hull). extrapolate needs proj_dim; the identity-at-init contract must still hold.
    try:
        KernelFiLM(embed_dim, 2 * H, extrapolate=True)  # no proj_dim -> reject
        raise AssertionError("extrapolate accepted without proj_dim")
    except ValueError:
        pass
    kfx = KernelFiLM(embed_dim, 2 * H, num_anchors=32, proj_dim=8, extrapolate=True).to(
        device
    )
    kfx.eval()
    assert torch.count_nonzero(kfx(ko)) == 0, (
        "extrapolate KernelFiLM must emit zeros at init"
    )
    assert "log_lambda" in kfx.state_dict()
    with torch.no_grad():
        kfx.values.copy_(torch.randn_like(kfx.values))
    lo, hi = kfx.values.min(dim=0).values, kfx.values.max(dim=0).values
    # small ridge -> the local plane escapes the hull on at least one coordinate (the whole point).
    with torch.no_grad():
        kfx.log_lambda.fill_(-6.0)
    out_lo = kfx(ko)
    assert torch.isfinite(out_lo).all()
    assert (out_lo < lo - 1e-3).any() or (out_lo > hi + 1e-3).any(), (
        "extrapolate KernelFiLM never left the convex hull -- it is not extrapolating"
    )
    # large ridge -> slope pinned to 0 -> recovers Nadaraya-Watson (`weights @ values`, in hull).
    with torch.no_grad():
        kfx.log_lambda.fill_(12.0)
    q = nn.functional.normalize(kfx.query_proj(ko), dim=-1)
    keys = nn.functional.normalize(kfx.keys, dim=-1)
    nw = (
        torch.softmax((q @ keys.t()) / kfx.log_tau.exp().clamp_min(kfx.tau_min), -1)
        @ kfx.values
    )
    assert torch.allclose(kfx(ko), nw, atol=1e-3), (
        "large ridge must recover Nadaraya-Watson"
    )

    # it installs in FlowCellModel via `ko_conditioner`, replacing the ko MLP under FiLM.
    film_kernel = FlowCellModel(
        t_processor_args=_args(t_dim, H),
        exp_processor_args=_args(n_genes, H),
        fused_processor_args=_args(H, n_genes, hidden=256),
        fusion_type="film",
        ko_conditioner=KernelFiLM(embed_dim, 2 * H, num_anchors=32),
        time_embedding_dim=t_dim,
    ).to(device)
    film_kernel.eval()
    v_kernel = film_kernel(t, x_t, ko)
    assert v_kernel.shape == (batch, n_genes) and torch.isfinite(v_kernel).all()
    assert (v_kernel < 0).any(), "kernel-FiLM velocity never negative"
    assert film_kernel.sample(x_t, ko, num_steps=4).shape == (batch, n_genes)

    # identity-init on the state: zero values (fresh KernelFiLM) + zeroed time branch must
    # leave the state untouched, exactly as the MLP FiLM head does.
    with torch.no_grad():
        film_kernel.t_processor.output_projection.weight.zero_()
        film_kernel.t_processor.output_projection.bias.zero_()
        expected_kernel = film_kernel.fused_processor(film_kernel.exp_processor(x_t))
    assert torch.allclose(film_kernel(t, x_t, ko), expected_kernel, atol=1e-6), (
        "zero-value KernelFiLM must be identity on the state"
    )

    # exactly one of ko_processor_args / ko_conditioner: both, or neither, must raise.
    for bad in (
        dict(
            ko_processor_args=_args(embed_dim, 2 * H),
            ko_conditioner=KernelFiLM(embed_dim, 2 * H),
        ),
        dict(),
    ):
        try:
            FlowCellModel(
                t_processor_args=_args(t_dim, H),
                exp_processor_args=_args(n_genes, H),
                fused_processor_args=_args(H, n_genes, hidden=256),
                fusion_type="film",
                time_embedding_dim=t_dim,
                **bad,
            )
            raise AssertionError("expected ValueError for ko branch specification")
        except ValueError:
            pass

    # 13. FiLMProcessingNN + film_deep: per-layer conditioning, generative-model style.
    Hd, Ld = 32, 3
    cond = KernelFiLM(input_size=embed_dim, output_size=2 * Hd * Ld, num_anchors=16)
    trunk = FiLMProcessingNN(
        input_size=n_genes,
        hidden_size=Hd,
        num_hidden_layers=Ld,
        output_size=n_genes,
        conditioner=cond,
    ).to(device)
    trunk.eval()

    t_emb_test = torch.randn(batch, Hd, device=device)
    out_trunk = trunk(x_t, ko, t_emb_test)
    assert out_trunk.shape == (batch, n_genes) and torch.isfinite(out_trunk).all()

    # zero-value conditioner => the perturbation contributes nothing => output is invariant to
    # which perturbation is passed. (The trunk itself is not identity -- it is a decoder.)
    ko2 = torch.randn(batch, embed_dim, device=device)
    assert torch.allclose(trunk(x_t, ko, t_emb_test), trunk(x_t, ko2, t_emb_test)), (
        "zero-value conditioner must make the trunk ignore the perturbation at init"
    )
    # entropy is mirrored from the conditioner for logging.
    assert 0.0 <= float(trunk.last_entropy) <= 1.0 + 1e-5, trunk.last_entropy

    # once the conditioner's values move, different perturbations diverge.
    with torch.no_grad():
        cond.values.copy_(torch.randn_like(cond.values))
    assert not torch.allclose(
        trunk(x_t, ko, t_emb_test), trunk(x_t, ko2, t_emb_test)
    ), "a non-trivial conditioner must make the trunk depend on the perturbation"

    # it installs in FlowCellModel as fusion_type='film_deep'.
    deep = FlowCellModel(
        t_processor_args=_args(t_dim, Hd),
        fusion_type="film_deep",
        trunk=FiLMProcessingNN(
            input_size=n_genes,
            hidden_size=Hd,
            num_hidden_layers=Ld,
            output_size=n_genes,
            conditioner=KernelFiLM(embed_dim, 2 * Hd * Ld, num_anchors=16),
        ),
        time_embedding_dim=t_dim,
    ).to(device)
    deep.eval()
    v_deep = deep(t, x_t, ko)
    assert v_deep.shape == (batch, n_genes) and torch.isfinite(v_deep).all()
    assert (v_deep < 0).any(), "film_deep velocity never negative"
    assert deep.sample(x_t, ko, num_steps=4).shape == (batch, n_genes)

    # validation: conditioner width must be 2*H*L; film_deep needs a trunk; time width must
    # match the trunk hidden width.
    try:
        FiLMProcessingNN(
            n_genes, Hd, Ld, n_genes, KernelFiLM(embed_dim, 2 * Hd * Ld + 1)
        )
        raise AssertionError("accepted conditioner width != 2*H*L")
    except ValueError:
        pass
    try:
        FlowCellModel(t_processor_args=_args(t_dim, Hd), fusion_type="film_deep")
        raise AssertionError("film_deep accepted without a trunk")
    except ValueError:
        pass
    try:
        FlowCellModel(
            t_processor_args=_args(t_dim, Hd + 1),  # != trunk hidden
            fusion_type="film_deep",
            trunk=FiLMProcessingNN(
                n_genes, Hd, Ld, n_genes, KernelFiLM(embed_dim, 2 * Hd * Ld)
            ),
            time_embedding_dim=t_dim,
        )
        raise AssertionError("film_deep accepted t.output != trunk.hidden_size")
    except ValueError:
        pass

    print(f"flow_model smoke test passed on {device}")
    print(f"  velocity {tuple(v.shape)}, range [{v.min():.3f}, {v.max():.3f}]")
    print(
        f"  time-embedding channel std: min {channel_std.min():.4f}, max {channel_std.max():.4f}"
    )
    print(
        f"  learnable table: {n_table} x {embed_dim}, "
        f"{sum(p.numel() for p in learn.parameters()):,} params"
    )
