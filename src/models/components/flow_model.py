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
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_anchors: int = 256,
        temperature: float = 0.1,
        tau_min: float = 1e-2,
        learn_temperature: bool = True,
    ) -> None:
        super().__init__()

        if num_anchors < 1:
            raise ValueError(f"num_anchors must be >= 1, got {num_anchors}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.input_size = input_size
        self.output_size = output_size
        self.num_anchors = num_anchors
        self.tau_min = tau_min

        # Keys tile the embedding manifold; random init is fine because softmax makes the
        # map smooth in `e` regardless of where the keys sit, and they receive gradient.
        self.keys = nn.Parameter(torch.randn(num_anchors, input_size))
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

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Map embeddings to FiLM params via kernel regression over the anchors.

        Args:
            e (torch.Tensor): Perturbation embeddings, shape ``(B, input_size)``. A zero row
                (an unannotated gene in the GO parquet) normalises to zero, giving uniform
                weights and hence the mean anchor response -- a sensible blind fallback.

        Returns:
            torch.Tensor: FiLM params, shape ``(B, output_size)``.
        """
        e = nn.functional.normalize(e, dim=-1)
        keys = nn.functional.normalize(self.keys, dim=-1)
        tau = self.log_tau.exp().clamp_min(self.tau_min)

        weights = torch.softmax((e @ keys.t()) / tau, dim=-1)  # (B, K)

        with torch.no_grad():
            p = weights.clamp_min(1e-12)
            entropy = -(p * p.log()).sum(-1) / math.log(self.num_anchors)
            self.last_entropy = entropy.mean()

        return weights @ self.values  # (B, output_size)


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
    fusion_type : {"sum", "product", "concat", "film"}
        How the three branches combine. ``sum``/``product`` require all three branches to
        share one width and combine them elementwise; ``concat`` stacks them. ``film`` is
        different in kind: the perturbation stops being a branch that is *added* and becomes
        a FiLM operator that *modulates the state*. ``ko_processor`` then emits ``[gamma;
        beta]`` (width ``2H``) and the fusion is ``(1 + gamma) * exp_processed + beta``,
        with time added afterward. The ``1 +`` makes it identity at init (gamma, beta ~ 0
        from the untrained linear head), so the state passes through untouched and the
        perturbation is learned as a deviation from it. This directly targets perturbation
        specificity: an additive perturbation signal averages away over cells and the model
        collapses to the mean profile, whereas a multiplicative operator cannot.
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
        exp_processor_args: dict[str, Any],
        fused_processor_args: dict[str, Any],
        fusion_type: Literal["sum", "product", "concat", "film"],
        ko_processor_args: dict[str, Any] | None = None,
        ko_conditioner: nn.Module | None = None,
        gene_embedding: nn.Module | None = None,
        time_embedding_dim: int = 128,
        time_embedding_scale: float = 1000.0,
    ) -> None:
        super().__init__()

        # The perturbation branch is either an MLP built from `ko_processor_args` (the
        # historical path) or an injected module such as `KernelFiLM` passed as
        # `ko_conditioner`. Exactly one; both interfaces are `(B, in) -> (B, out)` with an
        # `.output_size`, so everything downstream is identical.
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

        # Identity keeps the historical contract: `ko` arrives as the precomputed
        # vector. It holds no parameters, so checkpoints from before this argument
        # existed still load unchanged.
        self.gene_embedding = (
            nn.Identity() if gene_embedding is None else gene_embedding
        )

        self.time_embedding = SinusoidalTimeEmbedding(
            dim=time_embedding_dim, scale=time_embedding_scale
        )
        self.t_processor = ProcessingNN(**t_processor_args)
        self.ko_processor = (
            ko_conditioner
            if ko_conditioner is not None
            else ProcessingNN(**ko_processor_args)
        )
        self.exp_processor = ProcessingNN(**exp_processor_args)
        self.fused_processor = ProcessingNN(**fused_processor_args)
        self.fusion = fusion_type

    def forward(
        self, t: torch.Tensor, x_t: torch.Tensor, ko: torch.Tensor
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
        t_processed = self.t_processor(self.time_embedding(t))
        ko_processed = self.ko_processor(self.gene_embedding(ko))
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
        self, x: torch.Tensor, ko: torch.Tensor, t_start: float, t_end: float
    ) -> torch.Tensor:
        """Take one explicit Euler step of the ODE ``dx/dt = v(t, x, ko)``.

        Args:
            x (torch.Tensor): Current state, shape ``(B, n_genes)``.
            ko (torch.Tensor): Perturbation, as accepted by :meth:`forward`.
            t_start (float): Time at the start of the step.
            t_end (float): Time at the end of the step.

        Returns:
            torch.Tensor: State at ``t_end``.
        """
        t = torch.full((x.shape[0], 1), float(t_start), device=x.device, dtype=x.dtype)

        return x + (t_end - t_start) * self.forward(t, x, ko)

    @torch.no_grad()
    def sample(
        self, x0: torch.Tensor, ko: torch.Tensor, num_steps: int = 4
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
            x = self.step(x, ko, i * dt, (i + 1) * dt)

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

    print(f"flow_model smoke test passed on {device}")
    print(f"  velocity {tuple(v.shape)}, range [{v.min():.3f}, {v.max():.3f}]")
    print(
        f"  time-embedding channel std: min {channel_std.min():.4f}, max {channel_std.max():.4f}"
    )
    print(
        f"  learnable table: {n_table} x {embed_dim}, "
        f"{sum(p.numel() for p in learn.parameters()):,} params"
    )
