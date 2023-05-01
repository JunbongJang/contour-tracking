'''
Junbong Jang
4/30/2023

Modified Tensorflow implementation of Neural Matching Field (NeMF) from https://github.com/KU-CVLAB/NeMF

'''
import tensorflow as tf
from einops import rearrange


def MLP_score(cost, src_coord, trg_coord, mlp):
    """
    Arguments:
        cost: B H_s H_t
        src_feat: B C H W
        trg_feat: B C H W
        src_coord: B K N C
        trg_coord: B K N C
    Returns:
        ... D
    """

    B, N, K, _ = src_coord.shape
    
    src_coord_embedded = self.pos_embed(src_coord)
    trg_coord_embedded = self.pos_embed(trg_coord)
    # cost = self.cost_embedding(cost)
    _, C, _, _, H_t, W_t = cost.shape
    
    cost = rearrange(cost, 'B C H_s H_t -> B (C H_t) H_s')
    cost = bilinear_sampler_1d(cost, src_coord[:, :, 0], src_coord[:, :, 1])
    cost = rearrange(cost, 'B (C H_t) N K -> (B N K) C H_t', C=C, H_t=H_t, W_t=W_t)
    trg_coord = rearrange(trg_coord, 'B N K C -> (B N K) () () C')
    cost = bilinear_sampler_1d(cost, trg_coord[:, :, 0], trg_coord[:, :, 1])
    cost = cost.view(B, N, K, C)

    cost = mlp(p=tf.concat([src_coord_embedded, trg_coord_embedded], axis=-1), c_plane=cost)

    return cost


# def embedding(x):
#     """
#     Args:
#         x: tensor of shape [..., dim]
#     Returns:
#         embedding: a harmonic embedding of `x`
#             of shape [..., (n_harmonic_functions * 2 + int(append_input)) * dim]
#     """
#     # x = (x + 1) / 2
#     x = x * 2 + 3
#     embed = (x[..., None] * self._frequencies).reshape(*x.shape[:-1], -1)
#     embed = torch.cat(
#         (embed.sin(), embed.cos(), x)
#         if self.append_input
#         else (embed.sin(), embed.cos()),
#         dim=-1,
#     )
#     return embed



# class HarmonicEmbedding(torch.nn.Module):
#     def __init__(
#         self,
#         n_harmonic_functions: int = 6,
#         omega_0: float = 1.0,
#         logspace: bool = False,
#         append_input: bool = False,
#     ) -> None:
#         """
#         Given an input tensor `x` of shape [minibatch, ... , dim],
#         the harmonic embedding layer converts each feature
#         (i.e. vector along the last dimension) in `x`
#         into a series of harmonic features `embedding`,
#         where for each i in range(dim) the following are present
#         in embedding[...]:
#             ```
#             [
#                 sin(f_1*x[..., i]),
#                 sin(f_2*x[..., i]),
#                 ...
#                 sin(f_N * x[..., i]),
#                 cos(f_1*x[..., i]),
#                 cos(f_2*x[..., i]),
#                 ...
#                 cos(f_N * x[..., i]),
#                 x[..., i],              # only present if append_input is True.
#             ]
#             ```
#         where N corresponds to `n_harmonic_functions-1`, and f_i is a scalar
#         denoting the i-th frequency of the harmonic embedding.
#         If `logspace==True`, the frequencies `[f_1, ..., f_N]` are
#         powers of 2:
#             `f_1, ..., f_N = 2**torch.arange(n_harmonic_functions)`
#         If `logspace==False`, frequencies are linearly spaced between
#         `1.0` and `2**(n_harmonic_functions-1)`:
#             `f_1, ..., f_N = torch.linspace(
#                 1.0, 2**(n_harmonic_functions-1), n_harmonic_functions
#             )`
#         Note that `x` is also premultiplied by the base frequency `omega_0`
#         before evaluating the harmonic functions.
#         Args:
#             n_harmonic_functions: int, number of harmonic
#                 features
#             omega_0: float, base frequency
#             logspace: bool, Whether to space the frequencies in
#                 logspace or linear space
#             append_input: bool, whether to concat the original
#                 input to the harmonic embedding. If true the
#                 output is of the form (x, embed.sin(), embed.cos()
#         """
#         super().__init__()

#         if logspace:
#             frequencies = 2.0 ** torch.arange(
#                 n_harmonic_functions,
#                 dtype=torch.float32,
#             )
#         else:
#             frequencies = torch.linspace(
#                 1.0,
#                 2.0 ** (n_harmonic_functions - 1),
#                 n_harmonic_functions,
#                 dtype=torch.float32,
#             )

#         self.register_buffer("_frequencies", frequencies * omega_0, persistent=False)
#         self.append_input = append_input

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: tensor of shape [..., dim]
#         Returns:
#             embedding: a harmonic embedding of `x`
#                 of shape [..., (n_harmonic_functions * 2 + int(append_input)) * dim]
#         """
#         # x = (x + 1) / 2
#         x = x * 2 + 3
#         embed = (x[..., None] * self._frequencies).reshape(*x.shape[:-1], -1)
#         embed = torch.cat(
#             (embed.sin(), embed.cos(), x)
#             if self.append_input
#             else (embed.sin(), embed.cos()),
#             dim=-1,
#         )
#         return embed