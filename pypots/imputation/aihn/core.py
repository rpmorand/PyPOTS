"""
The core wrapper assembles the submodules of AIHN imputation and forecasting model
and takes over the forward progress of the algorithm.

"""

# Created by Knut Stroemmen <knut.stroemmen@unibe.com> and Rafael Morand <rafael.morand@unibe.ch>
# License: BSD-3-Clause

from typing import Callable

import torch
import torch.nn as nn

from ...nn.modules.aihn import (
    AIHNEmbedding, 
    FullAttention, 
    AttentionLayer, 
    Encoder, 
    EncoderLayer
)
from ...utils.metrics import calc_mae


class _AIHN(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """
    def __init__(
        self,
        n_layers: int,
        n_steps: int,
        main_freq: int,
        n_features: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
        customized_loss_func: Callable = calc_mae,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.n_features = n_features
        self.main_freq = main_freq
        self.d_k = d_k
        self.d_v = d_v
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.customized_loss_func = customized_loss_func
        
        self.output_size = main_freq
        n_max_steps = 20 * n_steps * n_features
        
        # Embedding
        self.embedding = AIHNEmbedding(self.main_freq, d_model, with_pos=True, n_max_steps=n_max_steps, dropout=dropout)
        
        # Encoder
        attn_layers = [
                EncoderLayer(
                    AttentionLayer(FullAttention(attention_dropout=attn_dropout, 
                                                 output_attention=True), 
                                   d_model, n_heads, d_k, d_v),
                    d_model, d_ffn, dropout=attn_dropout) for _ in range(n_layers)
            ]
        self.encoder = Encoder(attn_layers=attn_layers, norm_layer=torch.nn.LayerNorm(d_model))
        
        # Output projection
        self.output_projection = nn.Linear(d_model, self.output_size, bias=True)


    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        X_emb = self.embedding(X)
        
        # Transformer encoder processing
        enc_output, attns = self.encoder(X_emb)

        # project the representation from the d_model-dimensional space to the original data space for output
        reconstruction = self.output_projection(enc_output)

        # replace the observed part with values from X
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction

        # ensemble the results as a dictionary for return
        results = {
            "imputed_data": imputed_data,
        }

        # if in training mode, return results with losses
        if training:
            X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            results["loss"] = self.customized_loss_func(reconstruction, X_ori, indicating_mask)

        return results