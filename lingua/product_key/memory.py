# Copyright (c) Meta Platforms, Inc. and affiliates.
from logging import getLogger
import math
from typing import Optional, List
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.parallel import parallelize_module

from .colwise_embedding_bag import ColwiseEmbeddingBag, xFormerEmbeddingBag

logger = getLogger()


@dataclass
class ProductKeyArgs:
    is_enabled: bool = False
    layers: str = (
        ""  # Which layers to have the memory with product key on (example "6,12")
    )
    mem_n_keys: int = 1024  # The number of keys in the memory.
    mem_heads: int = 4  # Number of memory reading heads
    mem_knn: int = 32  # Number of memory slots to read / update - k-NN to the query
    mem_share_values: bool = True  # Share values across memories
    mem_k_dim: int = 512  # Memory keys dimension
    mem_v_dim: int = -1  # Memory values dimension (-1 for automatic output dimension)
    swilu_projection: bool = True
    value_fixed_lr: Optional[
        float
    ] = 0.001  # The learning rate for the value of the PK network
    mem_gated: bool = False
    peer_variant: bool = False


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class MultipleHashingMemory(nn.Module):
    def __init__(self, input_dim, output_dim, productkey_args: List[ProductKeyArgs]):
        super().__init__()
        self.memories = nn.ModuleList()
        for args in productkey_args:
            self.memories.append(
                HashingMemory(
                    input_dim,
                    output_dim,
                    mem_n_keys=args.mem_n_keys,
                    mem_heads=args.mem_heads,
                    mem_knn=args.mem_knn,
                    mem_share_values=args.mem_share_values,
                    mem_k_dim=args.mem_k_dim,
                    mem_v_dim=args.mem_v_dim,
                    swilu_projection=args.swilu_projection,
                    value_fixed_lr=args.value_fixed_lr
                    if args.value_fixed_lr is not None
                    else 0.001,
                    mem_gated=args.mem_gated,
                    peer_variant=args.peer_variant,
                )
            )

    def forward(self, input):
        """Combine multiple HashingMemory modules by pooling their candidates and selecting global top-k.

        Assumptions/scope:
        - All sub-memories share the same value table (mem_share_values=True) and are PK (not PEER).
        - heads, knn, k_dim, and size are identical across memories.
        """
        assert len(self.memories) > 0, "No memories configured"
        mem0 = self.memories[0]
        # Restrict to PK variant and shared values for now
        for m in self.memories:
            assert not m.use_peer_variant, "MultipleHashingMemory does not support PEER variant yet"
        # Compute queries and candidates per memory uniformly
        B, T, _ = input.shape
        cand_scores_list = []
        cand_indices_list = []
        x0 = None
        bs = None
        h = mem0.heads
        knn = mem0.knn
        kdim = mem0.k_dim
        for i, m in enumerate(self.memories):
            q, x_flat, bs_i, B_i, T_i = m.compute_query(input)
            if i == 0:
                x0 = x_flat
                bs = bs_i
                # Validate compatibility using the first memory as reference
                assert B_i == B and T_i == T
            # Validate compatibility across memories
            assert m.input_dim == mem0.input_dim
            assert m.output_dim == mem0.output_dim
            assert m.heads == h
            assert m.knn == knn
            assert m.k_dim == kdim
            assert m.size == mem0.size
            assert m.v_dim == mem0.v_dim
            # select all candidate pairs (~ knn^2) and reshape to (bs, h, C)
            scores_all, indices_all = m.select_candidates(q, knn)
            cand_scores_list.append(scores_all.view(bs_i, h, -1))
            cand_indices_list.append(indices_all.view(bs_i, h, -1))

        # Concatenate candidates across memories along candidate dimension
        all_scores = torch.cat(cand_scores_list, dim=2)
        all_indices = torch.cat(cand_indices_list, dim=2)

        # Global top-k across all memories per head
        top_scores, top_pos = torch.topk(all_scores, k=knn, dim=2, largest=True, sorted=True)
        top_indices = all_indices.gather(2, top_pos)

        # Use first memory to aggregate and project
        return mem0.aggregate_from_indices(x0, top_scores.view(bs * h, -1), top_indices.view(bs * h, -1), bs, B, T)

    def mp_parallelize(self, mesh, model_args, distributed_args, param_dtype):
        """Parallelize all underlying HashingMemory modules consistently.

        This delegates to each memory's mp_parallelize so that value tables
        get sharded/parallelized properly and shared when configured.
        """
        for m in self.memories:
            m.mp_parallelize(mesh, model_args, distributed_args, param_dtype)

    def reset_parameters(self, init_std=None, factor=1.0):
        for m in self.memories:
            m.reset_parameters(init_std=init_std, factor=factor)

class HashingMemory(nn.Module):

    VALUES = None
    EVAL_MEMORY = True

    def __init__(
        self,
        input_dim,
        output_dim,
        value_fixed_lr=0.001,  # added for xlformers and replaces mem_value_optimizer, set to None to use the same learning rate as the rest of the model
        # global parameters
        mem_k_dim=512,  # Memory keys dimension
        mem_v_dim=-1,  # Memory values dimension (-1 for automatic output dimension)
        mem_heads=4,  # Number of memory reading heads
        mem_knn=32,  # Number of memory slots to read / update - k-NN to the query
        mem_share_values=True,  # Share values across memories
        # keys
        mem_n_keys=1024,  # Number of keys
        # queries
        mem_query_bias=True,  # Query MLP bias
        mem_query_batchnorm=False,  # Query MLP batch norm
        # gating
        mem_gated=False,  # gated memory
        # values initialization
        # dropout
        mem_input_dropout=0.0,  # Input dropout
        mem_query_dropout=0.0,  # Query dropout
        mem_value_dropout=0.0,  # Value dropout
        # architecture
        peer_variant=False,  # Replaces the PK memory with the PEER variant (Parameter Efficient Expert Retrieval)
        swilu_projection=True,
    ):
        # Check parameters
        # even number of key dimensions for product quantization
        assert mem_k_dim >= 2

        # dropout
        assert 0 <= mem_input_dropout < 1
        assert 0 <= mem_query_dropout < 1
        assert 0 <= mem_value_dropout < 1

        # PEER variant
        assert not (
            peer_variant and mem_v_dim > 0
        ), f"Cannot use PEER variant with a value dimension different from the input dimension (mem_v_dim=-1)"

        # dimensions
        assert mem_k_dim % 2 == 0
        assert mem_heads >= 2

        # query batchnorm
        if mem_query_batchnorm:
            logger.warning(
                "WARNING: if you use batch normalization, be sure that you use batches of sentences with the same size at training time. Otherwise, the padding token will result in incorrect mean/variance estimations in the BatchNorm layer."
            )

        # initialize
        super().__init__()
        self.use_peer_variant = peer_variant

        # global parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        # number of indices / entries in the memory
        self.size = mem_n_keys**2
        self.k_dim = mem_k_dim

        self.v_dim = mem_v_dim if mem_v_dim > 0 else output_dim

        # values initialization
        self.swilu_proj = swilu_projection
        self.v_proj = mem_v_dim > 0 or self.swilu_proj
        self.heads = mem_heads
        self.knn = mem_knn

        # dropout
        self.input_dropout = mem_input_dropout
        self.query_dropout = mem_query_dropout
        self.value_dropout = mem_value_dropout

        # initialize keys
        self.keys = nn.Parameter(
            torch.empty(2 * self.heads * int(self.size**0.5), self.k_dim // 2)
        )

        # optionally use the same values for all memories
        self.mem_share_values = mem_share_values

        self.original = not self.mem_share_values or HashingMemory.VALUES is None

        # initialize the values
        if self.original:
            if not self.use_peer_variant:  # PK
                self.values = xFormerEmbeddingBag(self.size, self.v_dim)
                HashingMemory.VALUES = self.values
            else:  # PEER
                self.values_u = nn.Embedding(self.size, self.v_dim)
                self.values_v = nn.Embedding(self.size, self.v_dim)
                HashingMemory.VALUES = self.values_u, self.values_v
        else:
            if not self.use_peer_variant:  # PK
                self.values = None
            else:  # PEER
                self.values_u = None
                self.values_v = None
        self.value_fixed_lr = value_fixed_lr

        if self.v_proj:
            proj_input = mem_v_dim
            if self.swilu_proj and proj_input < 0:
                proj_input = output_dim
            self.value_proj = torch.nn.Linear(proj_input, output_dim)
        if self.swilu_proj:
            self.swilu_projection = torch.nn.Linear(self.input_dim, proj_input)
        # gated memory
        self.gating = None
        if mem_gated:
            self.gating = torch.nn.Linear(input_dim, 1)

        # query network
        # layer sizes / number of features
        l_sizes = (self.input_dim, self.heads * self.k_dim)

        self.query_proj = QueryMLP(
            self.input_dim,
            self.heads,
            self.k_dim,
            l_sizes,
            bias=mem_query_bias,
            batchnorm=mem_query_batchnorm,
        )

    def mp_parallelize(self, mesh, model_args, distributed_args, param_dtype):
        fsdp_config = dict(
            mp_policy=(
                MixedPrecisionPolicy(
                    param_dtype=param_dtype,
                    # reduce_dtype=torch.float32,
                    reduce_dtype=torch.bfloat16,
                )
            ),
            mesh=mesh["dp_replicate"],
        )
        # parallelize the module
        if distributed_args.memory_parallel_size > 1:
            assert (
                not self.use_peer_variant
            ), f"The PEER variant does not have a memory parallel implementation"
            if self.original:
                layer_plan = {"values": ColwiseEmbeddingBag()}
                parallelize_module(
                    self,
                    mesh["memory_parallel"],
                    layer_plan,
                )

        # share the parameters
        if self.original:
            if not self.use_peer_variant:
                self.values = fully_shard(
                    self.values, **fsdp_config, reshard_after_forward=False
                )
            else:
                self.values_u = fully_shard(
                    self.values_u, **fsdp_config, reshard_after_forward=False
                )
                self.values_v = fully_shard(
                    self.values_v, **fsdp_config, reshard_after_forward=False
                )
        if self.mem_share_values and self.original:
            if not self.use_peer_variant:
                HashingMemory.VALUES = self.values
            else:
                HashingMemory.VALUES = self.values_u, self.values_v
        if self.mem_share_values and not self.original:
            if not self.use_peer_variant:
                self.values = HashingMemory.VALUES
            else:
                self.values_u, self.values_v = HashingMemory.VALUES

    def reset_parameters(self, init_std=None, factor=1.0):
        # keys
        bound = 1 / math.sqrt(self.k_dim)
        nn.init.uniform_(self.keys, a=-bound, b=bound)
        # values
        if self.original:
            if not self.use_peer_variant:
                nn.init.normal_(self.values.weight, mean=0, std=self.v_dim**-0.5)
            else:
                nn.init.normal_(self.values_u.weight, mean=0, std=self.v_dim**-0.5)
                nn.init.normal_(self.values_v.weight, mean=0, std=self.v_dim**-0.5)
        # queries
        nn.init.xavier_uniform_(self.query_proj.query_mlps[0].weight)
        # value projection
        if self.v_proj:
            nn.init.normal_(self.value_proj.weight, mean=0, std=self.output_dim**-0.5)
        if self.swilu_proj:
            nn.init.normal_(
                self.swilu_projection.weight, mean=0, std=self.output_dim**-0.5
            )
        # fixed learning rate:
        if self.original:
            if self.use_peer_variant:
                for p in self.values_u.parameters():
                    p.fixed_lr = self.value_fixed_lr
                    p.pk_value_param = True
                for p in self.values_v.parameters():
                    p.fixed_lr = self.value_fixed_lr
                    p.pk_value_param = True
            else:
                for p in self.values.parameters():
                    p.fixed_lr = self.value_fixed_lr
                    p.pk_value_param = True
        if self.gating is not None:
            nn.init.normal_(self.gating.weight, mean=0, std=self.input_dim**-0.5)

    def forward(self, input):
        """
        Read from the memory.
        """
        query, input_flat, bs, B, T = self.compute_query(input)

        # get indices
        knn = self.knn
        scores, indices = self.get_indices(query, knn)  # (bs * heads, knn) ** 2

        return self.aggregate_from_indices(input_flat, scores, indices, bs, B, T)

    def compute_query(self, input: torch.Tensor):
        """Prepare flattened input and compute query embeddings for this memory.

        Returns: (query, input_flat, bs, B, T)
        """
        B, T, C = input.shape
        input_flat = input.view(-1, self.input_dim)
        assert input_flat.shape[-1] == self.input_dim
        prefix_shape = input_flat.shape[:-1]
        bs = int(np.prod(prefix_shape))
        input_flat = F.dropout(input_flat, p=self.input_dropout, training=self.training)
        query = self.query_proj(input_flat)
        query = F.dropout(query, p=self.query_dropout, training=self.training)
        assert query.shape == (bs * self.heads, self.k_dim)
        return query, input_flat, bs, B, T

    def aggregate_from_indices(self, input_flat: torch.Tensor, scores: torch.Tensor, indices: torch.Tensor, bs: int, B: int, T: int):
        """Aggregate values using selected indices and scores and apply projections/gating.

        Expects scores/indices shaped (bs*heads, knn).
        """
        knn = scores.size(-1)
        # store indices / scores (eval mode only - for usage statistics)
        if not self.training and HashingMemory.EVAL_MEMORY:
            self.last_indices = indices.view(bs, self.heads, knn).detach().cpu()
            self.last_scores = scores.view(bs, self.heads, knn).detach().cpu().float()

        # re-scoring
        scores = F.softmax(scores.float(), dim=-1).type_as(scores)

        # merge heads / knn (since we sum heads)
        indices = indices.view(bs, self.heads * knn)
        scores = scores.view(bs, self.heads * knn)

        if not self.use_peer_variant:
            output = self.values(indices, scores)
            if self.v_proj and not self.swilu_proj:
                output = self.value_proj(output)
            if self.swilu_proj:
                output = self.value_proj(output * F.silu(self.swilu_projection(input_flat)))
        else:
            u = self.values_u(indices)
            x = torch.einsum("bh, blh->bl", input_flat, u)
            x = F.gelu(x)
            v = self.values_v(indices)
            x = x * scores
            output = torch.einsum("bl, blh->bh", x, v)

        output = F.dropout(output, p=self.value_dropout, training=self.training)

        # reshape output to (B, T, v_dim)
        output = output.view(B, T, -1)
        if self.gating is not None:
            output = torch.sigmoid(self.gating(input_flat)).view(B, T, 1) * output
        return output

    def get_indices(self, query, knn):
        # Use select_candidates then pick the final top-k per head
        scores_all, indices_all = self.select_candidates(query, knn)
        bs = (scores_all.shape[0]) // self.heads
        # reshape to (bs, h, C)
        scores_all = scores_all.view(bs, self.heads, -1)
        indices_all = indices_all.view(bs, self.heads, -1)
        scores, best_positions = torch.topk(
            scores_all, k=knn, dim=2, largest=True, sorted=True
        )
        indices = indices_all.gather(2, best_positions)
        assert scores.shape == indices.shape == (bs, self.heads, knn)
        return scores.view(bs * self.heads, knn), indices.view(bs * self.heads, knn)

    def select_candidates(self, query, knn):
        """Return all knn^2 candidate scores and indices per query head before final top-k.

        Args:
            query: Tensor of shape (bs*heads, k_dim)
            knn: number of nearest keys per half to combine

        Returns:
            scores_all: (bs*heads, knn*knn)
            indices_all: (bs*heads, knn*knn)
        """
        assert query.dim() == 2 and query.size(1) == self.k_dim
        bs = len(query) // self.heads
        query = query.view(-1, self.heads, self.k_dim)
        half = self.k_dim // 2
        # keys : (heads, 2, n_keys, half)
        keys = self.keys.view(self.heads, 2, -1, half)
        keys1 = keys[:, 0, :, :]
        keys2 = keys[:, 1, :, :]
        n_keys = keys1.shape[1]

        # split query for product quantization
        q1 = query[:, :, :half]
        q2 = query[:, :, half:]

        # compute scores against sub-keys and keep top-k per half
        scores1 = torch.einsum("blh, lkh->blk", q1, keys1)
        scores2 = torch.einsum("blh, lkh->blk", q2, keys2)
        scores1, indices1 = scores1.topk(knn, dim=2, largest=True)
        scores2, indices2 = scores2.topk(knn, dim=2, largest=True)

        # Cartesian product of top candidates
        all_scores = (
            scores1.view(bs, self.heads, knn, 1).expand(bs, self.heads, knn, knn)
            + scores2.view(bs, self.heads, 1, knn).expand(bs, self.heads, knn, knn)
        ).reshape(bs, self.heads, -1)
        all_indices = (
            indices1.view(bs, self.heads, knn, 1).expand(bs, self.heads, knn, knn)
            * n_keys
            + indices2.view(bs, self.heads, 1, knn).expand(bs, self.heads, knn, knn)
        ).reshape(bs, self.heads, -1)

        return all_scores.view(bs * self.heads, -1), all_indices.view(bs * self.heads, -1)

class QueryMLP(nn.Module):
    def __init__(self, input_dim, heads, k_dim, sizes, bias=False, batchnorm=False):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.k_dim = k_dim
        self.sizes = sizes
        assert sizes[0] == input_dim
        assert sizes[-1] == (heads * k_dim)

        # MLPs
        sizes_ = list(sizes)
        sizes_[-1] = sizes_[-1]
        self.query_mlps = QueryMLP.mlp(sizes_, bias=bias, batchnorm=batchnorm)

    @staticmethod
    def mlp(sizes, bias=True, batchnorm=True):
        """
        Generate a feedforward neural network.
        """
        assert len(sizes) >= 2
        pairs = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        layers = []

        for i, (dim_in, dim_out) in enumerate(pairs):
            layers.append(nn.Linear(dim_in, dim_out, bias=bias))
            if batchnorm:
                layers.append(nn.BatchNorm1d(dim_out))
            if i < len(pairs) - 1:
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, input):
        """
        Compute queries using either grouped 1D convolutions or ModuleList + concat.
        """
        assert input.shape[-1] == self.input_dim
        input = (
            input.contiguous().view(-1, self.input_dim) if input.dim() > 2 else input
        )
        bs = len(input)

        outputs = [m(input) for m in self.query_mlps]
        query = torch.cat(outputs, 1) if len(outputs) > 1 else outputs[0]

        assert query.shape == (bs, self.heads * self.k_dim)
        return query.view(bs * self.heads, self.k_dim)
