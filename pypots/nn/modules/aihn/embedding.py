"""

"""

# Created by Knut Stroemmen <knut.stroemmen@unibe.com> and Rafael Morand <rafael.morand@unibe.ch>
# License: BSD-3-Clause

import torch
import torch.nn as nn


class AIHNEmbedding(nn.Module):
    """The embedding method for AIHN inspired by the iTransformer and main frequency-based imputation & forecasting model.

    Parameters
    ----------
    d_in :
        The input dimension.

    d_out :
        The output dimension.

    with_pos :
        Whether to add positional encoding.

    n_max_steps :
        The maximum number of steps.
        It only works when ``with_pos`` is True.

    dropout :
        The dropout rate.

    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        with_pos: bool,
        n_max_steps: int = 1000,
        dropout: float = 0,
    ):
        super().__init__()
        self.with_pos = with_pos
        self.dropout_rate = dropout  
        
        self.d_in = d_in
        self.d_out = d_out   
        
        self.embedding_layer = nn.Linear(d_in, d_out)
        self.position_enc = InvertedPositionalEncoding(d_out, n_positions=n_max_steps) if with_pos else None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, X):
        X_embedding = self.embedding_layer(X)
        if self.with_pos:
            X_embedding = self.position_enc(X_embedding)
        if self.dropout_rate > 0:
            X_embedding = self.dropout(X_embedding)
        return X_embedding    
    
    
class InvertedPositionalEncoding(nn.Module):
    """The original positional-encoding module for Transformer.

    Parameters
    ----------
    d_hid:
        The dimension of the hidden layer.

    n_positions:
        The max number of positions.

    """

    def __init__(self, d_hid: int, n_positions: int = 1000):
        super().__init__()
        pe = torch.zeros(n_positions, d_hid, requires_grad=False).float()
        position = torch.arange(0, n_positions).float().unsqueeze(1)
        div_term = (torch.arange(0, d_hid, 2).float() * -(torch.log(torch.tensor(10000)) / d_hid)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe = pe.unsqueeze(0).transpose(0, 1) #! Transpose the position encoding (opp. to the original w/o transpose)
        pe = pe.unsqueeze(0) #! Transpose the position encoding (opp. to the original w/o transpose)

        self.register_buffer("pe", pe)

    def forward(self, X: torch.Tensor, return_only_pos: bool = False) -> torch.Tensor:
        """Forward processing of the positional encoding module.

        Parameters
        ----------
        X:
            Input tensor.

        return_only_pos:
            Whether to return only the positional encoding.

        Returns
        -------
        If return_only_pos is True:
            pos_enc:
                The positional encoding.
        else:
            X_with_pos:
                Output tensor, the input tensor with the positional encoding added.
        """
        pos_enc = self.pe[:, :X.size(1)].clone().detach()
        return X + pos_enc
