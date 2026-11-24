# ============================================================
# GH07_model.py - LifeGraph model (GATv2 encoder + shared residual MLP + batched per-node DSM head)
# Code copied verbatim from GH7.3 cells 14 (model, build_batch), 16 (DSM loss) and 29 (survival function).
# Architecture is unchanged; GH07_train.ipynb and GH07_eval.ipynb import everything from here.
# ============================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


# ------------------------------------------------------------
# Model (GH7.3 cell 14)
# ------------------------------------------------------------
class FunctionalTimeEncoding(nn.Module):
    """Learnable Fourier-style time encoding for the years_before scalar."""
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.freq = nn.Parameter(torch.randn(dim // 2))
    def forward(self, t):
        t = t.unsqueeze(-1)
        x = self.freq * t
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class BatchedDSMHead(nn.Module):
    """Per-ICD Deep Survival Machines head, vectorized over N nodes.

    Equivalent to nn.ModuleList of N independent DeepSurvivalMachinesTorch(layers=[]),
    but runs as a single einsum instead of an N-way Python loop. Each ICD gets:
      - shapeg / scaleg: [hidden_dim, k] linear projections to k mixture params
      - gate:           [hidden_dim, k] linear projection to k mixture logits
      - shape / scale bias: [k] per-node bias added after activation
    Forward output: (shape, scale, logits) each of shape [B, N, k].
    """
    def __init__(self, hidden_dim, n_nodes, k=3, dist="Weibull", temp=1000.0):
        super().__init__()
        self.n_nodes = n_nodes
        self.k = k
        self.dist = dist
        self.temp = float(temp)

        # Per-node distribution bias parameters (auton-survival inits to -1 for Weibull, +1 for LogNormal)
        init_val = -1.0 if dist == "Weibull" else 1.0
        self.shape_bias = nn.Parameter(torch.full((n_nodes, k), init_val))
        self.scale_bias = nn.Parameter(torch.full((n_nodes, k), init_val))

        # Per-node linear projection weights and biases
        # Stored as [N, hidden_dim, k] so einsum 'bnd,ndk->bnk' applies the n-th matrix to h[:, n, :]
        self.shapeg_w = nn.Parameter(torch.zeros(n_nodes, hidden_dim, k))
        self.shapeg_b = nn.Parameter(torch.zeros(n_nodes, k))
        self.scaleg_w = nn.Parameter(torch.zeros(n_nodes, hidden_dim, k))
        self.scaleg_b = nn.Parameter(torch.zeros(n_nodes, k))
        # Gate has no bias in DSM (auton-survival uses bias=False)
        self.gate_w   = nn.Parameter(torch.zeros(n_nodes, hidden_dim, k))

        # Xavier init for projection weights
        nn.init.xavier_uniform_(self.shapeg_w)
        nn.init.xavier_uniform_(self.scaleg_w)
        nn.init.xavier_uniform_(self.gate_w)

        # DSM activation: SELU for Weibull, Tanh for LogNormal (matches auton-survival)
        if dist == "Weibull":
            self.act = nn.SELU()
        elif dist == "LogNormal":
            self.act = nn.Tanh()
        else:
            raise NotImplementedError(f"dist={dist} not supported")

    def forward(self, h):
        """
        h:        [B, N, hidden_dim] encoder output
        returns:  shape, scale, logits — each [B, N, k]
        shape_3d  =  log(k)        →  k      = exp(shape_3d)
        scale_3d  =  log(lambda)   →  lambda = exp(scale_3d)
        gate_3d   =  raw logits    →  pi     = softmax(gate_3d)  along K
        """
        # Per-node linear projections via einsum (one matmul per ICD, fused into a single op)
        shape_raw = torch.einsum("bnd,ndk->bnk", h, self.shapeg_w) + self.shapeg_b   # [B, N, k]
        scale_raw = torch.einsum("bnd,ndk->bnk", h, self.scaleg_w) + self.scaleg_b   # [B, N, k]
        logits    = torch.einsum("bnd,ndk->bnk", h, self.gate_w) / self.temp         # [B, N, k]

        # Activation + per-node distribution bias (matches DSM's shape/scale parameter additions)
        shape = self.act(shape_raw) + self.shape_bias.unsqueeze(0)                   # [B, N, k]
        scale = self.act(scale_raw) + self.scale_bias.unsqueeze(0)                   # [B, N, k]
        return shape, scale, logits


class DiseaseGATDSM(nn.Module):
    """GAT encoder + shared MLP residual blocks + batched per-ICD DSM head."""

    def __init__(self, in_dim, hidden_dim, edge_dim, n_nodes,
                 dsm_k=3, dsm_dist="Weibull", dsm_temp=1000.0, dsm_discount=1.0,
                 heads=4, dropout=0.2, use_time_encoding=True, time_dim=8, anc_idx=71, n_ancestry=4, anc_dim=4):
        super().__init__()
        self.n_nodes = n_nodes
        self.dsm_k = dsm_k
        self.dsm_dist = dsm_dist
        self.dsm_discount = float(dsm_discount)
        self.dropout = dropout
        self.use_time_encoding = use_time_encoding

        # Ancestry embedding: integer code at anc_idx -> learned anc_dim vector
        self.anc_idx = anc_idx
        self.anc_emb = nn.Embedding(n_ancestry, anc_dim)
        
        if use_time_encoding:
            self.time_enc = FunctionalTimeEncoding(time_dim)
            # Drop years_before scalar (+time_dim) and ancestry code (+anc_dim)
            effective_in_dim = in_dim - 1 + time_dim - 1 + anc_dim
        else:
            self.time_enc = None
            # Drop ancestry code (+anc_dim)
            effective_in_dim = in_dim - 1 + anc_dim
        self.effective_in_dim = effective_in_dim

        # GAT layer 1: effective_in_dim -> hidden_dim per head, concat heads
        self.gat1 = GATv2Conv(effective_in_dim, hidden_dim, heads=heads,
                              edge_dim=edge_dim, dropout=dropout, concat=True)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)

        # GAT layer 2: average heads for stable hidden_dim output
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads,
                              edge_dim=edge_dim, dropout=dropout, concat=False)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Shared MLP block 1 with pre-norm residual: h1 = h0 + MLP1(LN(h0))
        self.shared_norm1 = nn.LayerNorm(hidden_dim)
        self.shared_mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Shared MLP block 2 with pre-norm residual: h2 = h1 + MLP2(LN(h1))
        self.shared_norm2 = nn.LayerNorm(hidden_dim)
        self.shared_mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Feature residual: project encoded raw input to hidden_dim (bias=False, clean skip)
        self.residual_proj = nn.Linear(effective_in_dim, hidden_dim, bias=False)

        # Bound the encoder output magnitude before feeding into DSM's exp(exp(.)) heads
        self.head_norm = nn.LayerNorm(hidden_dim)

        # Batched per-ICD DSM head (one BatchedDSMHead replaces N separate DSM modules)
        self.dsm_head = BatchedDSMHead(
            hidden_dim=hidden_dim, n_nodes=n_nodes,
            k=dsm_k, dist=dsm_dist, temp=dsm_temp,
        )
        
        # Attention capture switch and storage; off by default to avoid memory overhead during training
        self.return_attn = False
        self.last_attn = None  

    def _encode_node_features(self, x):
        # x: [B, N, in_dim] = [obs, years_before, prs1-3, emb(64), age, sex, ancestry_code]
        # Ancestry code -> embedding vector, removed from the numeric features
        anc_vec = self.anc_emb(x[..., self.anc_idx].long())                          # [B, N, anc_dim]
        x_num   = torch.cat([x[..., :self.anc_idx], x[..., self.anc_idx + 1:]], dim=-1)

        if not self.use_time_encoding:
            return torch.cat([x_num, anc_vec], dim=-1)

        obs        = x_num[..., 0:1]                   # [B, N, 1]
        yrs_scalar = x_num[..., 1]                     # [B, N]
        rest       = x_num[..., 2:]                    # prs, emb, age, sex
        t_vec = self.time_enc(yrs_scalar) * obs        # zero out where not observed
        return torch.cat([obs, t_vec, rest, anc_vec], dim=-1)

    def encode(self, x, edge_index, edge_attr,
               codx_edge_index, codx_edge_attr, codx_edge_ptr, batch_sample_ids):
        """Return per-ICD representation tensor of shape [B, N, hidden_dim]."""
        # Encode raw features once; reuse as GAT input AND as feature residual source
        x_enc = self._encode_node_features(x)            # [B, N, effective_in_dim]
        B, N, F_in = x_enc.shape
        dev = x_enc.device

        # Flatten batch into a single disjoint graph by offsetting node indices
        x_flat = x_enc.reshape(B * N, F_in)
        offsets = (torch.arange(B, device=dev) * N).view(B, 1, 1)
        g_edge_index = (edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
        g_edge_attr  = edge_attr.repeat(B, 1)

        # Gather per-sample co-dx edges
        codx_parts_idx, codx_parts_attr = [], []
        for b, sid in enumerate(batch_sample_ids.tolist()):
            start = codx_edge_ptr[sid].item()
            end   = codx_edge_ptr[sid + 1].item()
            if end == start:
                continue
            local = codx_edge_index[:, start:end] + b * N
            codx_parts_idx.append(local)
            codx_parts_attr.append(codx_edge_attr[start:end])

        if codx_parts_idx:
            c_edge_index = torch.cat(codx_parts_idx, dim=1)
            c_edge_attr  = torch.cat(codx_parts_attr, dim=0)
            edge_index_b = torch.cat([g_edge_index, c_edge_index], dim=1)
            edge_attr_b  = torch.cat([g_edge_attr, c_edge_attr], dim=0)
        else:
            edge_index_b = g_edge_index
            edge_attr_b  = g_edge_attr

        # GAT message passing
        if self.return_attn:
            h, (attn_ei_1, attn_w_1) = self.gat1(
                x_flat, edge_index_b, edge_attr=edge_attr_b,
                return_attention_weights=True,
            )
        else:
            h = self.gat1(x_flat, edge_index_b, edge_attr=edge_attr_b)
            attn_ei_1, attn_w_1 = None, None
            
        h = F.elu(self.norm1(h))
        
        if self.return_attn:
            h, (attn_ei_2, attn_w_2) = self.gat2(
                h, edge_index_b, edge_attr=edge_attr_b,
                return_attention_weights=True,
            )
        else:
            h = self.gat2(h, edge_index_b, edge_attr=edge_attr_b)
            attn_ei_2, attn_w_2 = None, None
        
        h = F.elu(self.norm2(h))                         # [B*N, hidden_dim]
        
        if self.return_attn:
            self.last_attn = {
                "B": B, "N": N,
                "layer1_edge_index": attn_ei_1.detach().cpu(),
                "layer1_alpha":      attn_w_1.detach().cpu(),
                "layer2_edge_index": attn_ei_2.detach().cpu(),
                "layer2_alpha":      attn_w_2.detach().cpu(),
            }
    
        # Shared MLP block 1 with pre-norm gradient residual (per-token, B*N axis is batch)
        h = h + self.shared_mlp1(self.shared_norm1(h))   # [B*N, hidden_dim]

        # Shared MLP block 2 with pre-norm gradient residual
        h = h + self.shared_mlp2(self.shared_norm2(h))   # [B*N, hidden_dim]

        # Reshape back to [B, N, hidden_dim] so we can add the per-(B, N) feature residual
        h = h.view(B, N, -1)

        # Feature residual: inject encoded raw inputs into head input
        h = h + self.residual_proj(x_enc)                # [B, N, hidden_dim]

        # Final LN keeps magnitude bounded before DSM's double-exp Weibull head
        h = self.head_norm(h)                            # [B, N, hidden_dim]
        return h

    def forward(self, x, edge_index, edge_attr,
                codx_edge_index, codx_edge_attr, codx_edge_ptr, batch_sample_ids):
        """returns:  shape, scale, logits — each [B, N, k]
        shape_3d  =  log(k)        →  k      = exp(shape_3d)
        scale_3d  =  log(lambda)   →  lambda = exp(scale_3d)
        gate_3d   =  raw logits    →  pi     = softmax(gate_3d)  along K
        """
        h = self.encode(x, edge_index, edge_attr,
                        codx_edge_index, codx_edge_attr, codx_edge_ptr, batch_sample_ids)
        shape, scale, logits = self.dsm_head(h)
        return shape, scale, logits
    
    @torch.no_grad()
    def get_attention_matrix(self, layer=2, head_reduce="mean"):
        """Convert captured attention edges into dense [B, N, N] matrices.

        layer:       1 or 2 — which GAT layer's attention to return
        head_reduce: 'mean' averages over heads, 'none' keeps head dim ([B, heads, N, N])

        Convention (PyG): alpha[e] is the attention weight FROM source TO target,
        where edge_index[0] = source, edge_index[1] = target. The returned matrix
        A[i, j] is the attention weight on the edge i -> j.
        """
        assert self.last_attn is not None, "Run forward() with model.return_attn=True first."
        B, N = self.last_attn["B"], self.last_attn["N"]
        ei    = self.last_attn[f"layer{layer}_edge_index"]   # [2, E_total]
        alpha = self.last_attn[f"layer{layer}_alpha"]        # [E_total, heads]
        heads = alpha.shape[1]

        # Recover sample id and local node ids from the disjoint graph offsets
        src_global, dst_global = ei[0], ei[1]
        sample_id = src_global // N                          # same for src and dst within a sample
        src_local = src_global %  N
        dst_local = dst_global %  N

        # Allocate dense container; sparse-to-dense by scatter
        A = torch.zeros(B, heads, N, N)
        A[sample_id, :, src_local, dst_local] = alpha       # broadcast over heads dim

        if head_reduce == "mean":
            return A.mean(dim=1)                             # [B, N, N]
        return A                                             # [B, heads, N, N]


def build_batch(local_idx, orig_idx, x_t, t_t, e_t, y_bin_t, m_t, tte_t, dev):
    # local_idx: positions within the split tensor
    # orig_idx:  cohort-wide indices (for codx_edge_ptr lookup)
    x     = x_t[local_idx].to(dev)
    t_b   = t_t[local_idx].to(dev)              # [B, N] survival times
    e_b   = e_t[local_idx].to(dev)              # [B, N] event indicators
    y_bin = y_bin_t[local_idx]                  # CPU; only used for eval
    m     = m_t[local_idx].to(dev)              # [B, N] mask
    tte   = tte_t[local_idx]                    # CPU; only used for eval
    sids  = torch.as_tensor(orig_idx, dtype=torch.long)
    return x, t_b, e_b, y_bin, m, tte, sids


# ------------------------------------------------------------
# DSM loss and horizon risk (GH7.3 cell 16)
# ------------------------------------------------------------
def _weibull_log_terms(shape, scale, t):
    """Weibull log-density and log-survival per mixture component.
    DSM parameterizes k = exp(shape), lambda = exp(scale), so:
      log S_g(t) = -(exp(scale) * t) ** exp(shape)
      log f_g(t) = shape + scale + (exp(shape) - 1) * (scale + log t) + log S_g(t)
    """
    log_t = torch.log(t)                                          # [B, N, 1]
    exp_k = torch.exp(shape)                                      # [B, N, k], calculating exp(shape) = k
    exp_b = torch.exp(scale)                                      # [B, N, k], calculating exp(scale) = lambda
    log_S = -torch.pow(exp_b * t, exp_k)                          # [B, N, k], calculating log⁡S(t)=−(λt)^k
    log_f = shape + scale + (exp_k - 1.0) * (scale + log_t) + log_S
    return log_f, log_S


def _lognormal_log_terms(shape, scale, t):
    """LogNormal log-density and log-survival per mixture component.

    DSM: mu = shape, sigma = exp(scale).
    """
    log_t = torch.log(t)                                          # [B, N, 1]
    mu    = shape                                                 # [B, N, k]
    sigma = torch.exp(scale)                                      # [B, N, k]
    log_f = -scale - 0.5 * np.log(2 * np.pi) \
            - 0.5 * ((log_t - mu) / sigma).pow(2)
    # S(t) = 0.5 * (1 - erf((log_t - mu) / (sqrt(2) * sigma)))
    z = (log_t - mu) / (sigma * np.sqrt(2.0))
    S = 0.5 - 0.5 * torch.erf(z)
    log_S = torch.log(S.clamp(min=1e-20))
    return log_f, log_S


def batched_dsm_loss(shape, scale, logits, t, e, mask,
                     dist="Weibull", elbo=True, discount=1.0):
    """Batched DSM conditional NLL, masked over valid (sample, ICD) pairs.

    Args:
        shape, scale, logits: [B, N, k] from BatchedDSMHead.forward
        t:    [B, N] survival times (must be > 0)
        e:    [B, N] event indicators (0/1)
        mask: [B, N] valid flag (1 = include in loss)
    Returns:
        Scalar mean NLL.
    """
    # Clamp t away from 0; expand to [B, N, 1] to broadcast across k components
    t_safe = t.clamp(min=1e-4).unsqueeze(-1)

    if dist == "Weibull":
        log_f_g, log_S_g = _weibull_log_terms(shape, scale, t_safe)
    elif dist == "LogNormal":
        log_f_g, log_S_g = _lognormal_log_terms(shape, scale, t_safe)
    else:
        raise NotImplementedError(f"dist={dist}")

    if elbo:
        # ELBO mixture aggregation: Σ_g π_g * log(.)
        pi = torch.softmax(logits, dim=-1)                        # [B, N, k]
        log_S = (pi * log_S_g).sum(dim=-1)                        # [B, N]
        log_f = (pi * log_f_g).sum(dim=-1)                        # [B, N]
    else:
        # Exact log-likelihood: logsumexp(log π_g + log(.))
        log_pi = torch.log_softmax(logits, dim=-1)
        log_S = torch.logsumexp(log_pi + log_S_g, dim=-1)
        log_f = torch.logsumexp(log_pi + log_f_g, dim=-1)

    # Per-entry log-likelihood: events use log f, censored use log S (with discount)
    # This is per-entry loss
    per_entry_ll = e * log_f + discount * (1.0 - e) * log_S       # [B, N]

    # Calculate avg loss in this batch
    return -(per_entry_ll * mask).sum() / mask.sum().clamp(min=1.0)


def batched_event_prob_at_horizon(shape, scale, logits, horizon, dist="Weibull"):
    """Compute P(event by horizon) = 1 - S(horizon) per (sample, ICD), vectorized.
    Args:
        shape, scale, logits: [B, N, k]
        horizon: scalar in same units as t (years)
    Returns:
        event_p: [B, N] tensor in [0, 1]
    """
    t = shape.new_full((1, 1, 1), float(horizon)).clamp(min=1e-4)
    
    if dist == "Weibull":
        _, log_S = _weibull_log_terms(shape, scale, t)          # [B, N, k]
    elif dist == "LogNormal":
        _, log_S = _lognormal_log_terms(shape, scale, t)
    else:
        raise NotImplementedError(f"dist={dist}")

    # Mixture survival in log-space for numerical stability
    # This is calculating ture mixture survival model，not ELBO mixture
    log_pi = torch.log_softmax(logits, dim=-1)
    log_S = torch.logsumexp(log_pi + log_S, dim=-1)             # [B, N]
    return (1.0 - torch.exp(log_S)).clamp(0.0, 1.0)


# ------------------------------------------------------------
# Mixture survival curves on a time grid (GH7.3 cell 29)
# ------------------------------------------------------------
def batched_survival_function(
    shape,
    scale,
    logits,
    time_grid,
    dist="Weibull",
):
    """
    Compute mixture survival curves S(t) for all samples and ICD nodes.

    Parameters
    ----------
    shape : torch.Tensor
        DSM shape parameters, shape [B, N, K].

        Weibull:
            k = exp(shape)

        LogNormal:
            mu = shape

    scale : torch.Tensor
        DSM scale parameters, shape [B, N, K].

        Weibull:
            lambda = exp(scale)

        LogNormal:
            sigma = exp(scale)

    logits : torch.Tensor
        Mixture logits, shape [B, N, K].

    time_grid : torch.Tensor or array-like
        Evaluation times, shape [T], in the same units as training times.

    dist : str
        "Weibull" or "LogNormal".

    Returns
    -------
    survival : torch.Tensor
        Mixture survival probabilities, shape [B, N, T].
    """

    # Ensure time_grid is on the same device and dtype as model outputs
    time_grid = torch.as_tensor(
        time_grid,
        dtype=shape.dtype,
        device=shape.device,
    )

    # [1, 1, 1, T]
    # Do not need a positive clamp for Weibull at t=0, but LogNormal
    # requires log(t), so it is handled separately below.
    t = time_grid.view(1, 1, 1, -1)

    # Mixture log-weights: [B, N, K, 1]
    log_pi = torch.log_softmax(
        logits,
        dim=-1,
    ).unsqueeze(-1)

    # DSM parameters: [B, N, K, 1]
    shape_4d = shape.unsqueeze(-1)
    scale_4d = scale.unsqueeze(-1)

    if dist == "Weibull":
        # DSM parameterization:
        # k      = exp(shape)
        # lambda = exp(scale)
        #
        # log S_g(t) = -(lambda * t)^k

        k = torch.exp(shape_4d)
        lam = torch.exp(scale_4d)

        # t=0 gives log S_g(0)=0 exactly
        log_S_g = -torch.pow(
            lam * t.clamp_min(0.0),
            k,
        )

    elif dist == "LogNormal":
        # DSM parameterization:
        # mu    = shape
        # sigma = exp(scale)

        mu = shape_4d
        sigma = torch.exp(scale_4d)

        # lognormal survival at t=0 is exactly 1
        positive_t = t > 0

        # Avoid log(0) internally
        t_safe = t.clamp_min(1e-12)
        log_t = torch.log(t_safe)

        z = (log_t - mu) / (
            sigma * np.sqrt(2.0)
        )

        S_g_positive = 0.5 - 0.5 * torch.erf(z)

        # At t=0, S_g(0)=1
        S_g = torch.where(
            positive_t,
            S_g_positive,
            torch.ones_like(S_g_positive),
        )

        log_S_g = torch.log(
            S_g.clamp_min(1e-20)
        )

    else:
        raise NotImplementedError(f"dist={dist}")

    # Exact mixture survival:
    #
    # S(t) = sum_g pi_g S_g(t)
    #
    # In log-space:
    # log S(t) = logsumexp(log pi_g + log S_g(t))
    #
    # Result: [B, N, T]
    log_S = torch.logsumexp(
        log_pi + log_S_g,
        dim=2,
    )

    survival = torch.exp(log_S).clamp(
        min=0.0,
        max=1.0,
    )

    return survival
