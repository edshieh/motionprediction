from soft_moe_pytorch import SoftMoE
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn as nn
import os
from torch.nn import LayerNorm
from torch.nn import MultiheadAttention
from torch.nn.init import xavier_uniform_

from fairmotion.models.SpatioTemporalTransformer import PositionalEncodingST, convert_on_device
import random

# Set environment variable for MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

#####
# For the following shape dimensions we use:
# B: Batch size
# T: Input sequence length
# S: Number of joints (24)
# E: Angle representation dimension (3 for aa)
# H: Hidden dimension of feed forward linear layer in attention layers
#####

class SpatialTemporalEncoderLayer(nn.Module):
    def __init__(self, ninp, num_heads, num_experts, dropout, device, use_double=True):
        super(SpatialTemporalEncoderLayer, self).__init__()
        self.device = device
        self.use_double = use_double
        self.SpatialMultiheadAttention = MultiheadAttention(ninp, num_heads, dropout)
        self.TemporalMultiheadAttention = MultiheadAttention(ninp, num_heads, dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.norm_1 = LayerNorm(ninp)
        self.norm_2 = LayerNorm(ninp)
        self.norm_3 = LayerNorm(ninp)

        self.moe = SoftMoE(
            dim = ninp*24,
            seq_len = num_experts,
            num_experts = num_experts
        )

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            convert_on_device(mask, self.device, self.use_double)
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, x):
        B, T, S, E = x.shape # B x T x S x E
        tx = torch.transpose(x, 0, 1) # T x B x S x E
        tx = torch.reshape(tx, (T, B*S, E)) # T x (B*S) x E
        # tx: input to temporal multihead (T, B*S, E)
        # t_mask :mask for temporal attention
        t_mask = self._generate_square_subsequent_mask(T).to(device=x.device, dtype=torch.float32)
        if self.use_double:
            t_mask = t_mask.double()
        tm, _ = self.TemporalMultiheadAttention(tx, tx, tx, attn_mask= t_mask) # T x (B*S) x E
        tm = self.dropout_1(tm)
        tm = self.norm_1(tm + tx)
        tm = torch.transpose(tm, 0, 1) # (B*S) x T x E
        tm = torch.reshape(tm, (B, S, T, E)) # B x S x T x E
        tm = torch.transpose(tm, 1, 2) # B x T x S x E

        sx = torch.reshape(x, (B * T, S, E)) # (B*T) x S x E
        sx = torch.transpose(sx, 0, 1)# S x (B*T) x E
        # sx: input to spatial multihead (S, B*T, E)
        sm, _ = self.SpatialMultiheadAttention(sx, sx, sx) # S x (B*T) x E
        sm = self.dropout_2(sm)
        sm = self.norm_2(sm + sx)
        sm = torch.transpose(sm, 0, 1) # (B*T) x S x E
        sm = torch.reshape(sm, (B, T, S, E)) # B x T x S x E

        # add temporal + spatial
        input = tm + sm # B x T x S x E

        #reshape so that it fits into MOE
        input_reshaped = torch.reshape(input, (B, T, S*E)) # B x T x (S x E)
        # moe block
        xx = self.moe(input_reshaped) # B x T x (S x E)
        # undo reshape
        xx = torch.reshape(xx, (B, T, S, E)) # B x T x S x E
        # dropout
        xx = self.dropout_3(xx)
        # add residual connection and norm
        output = self.norm_3(xx + input)
        return output




class moe(nn.Module):
    def __init__(
        self, ntoken, ninp, num_heads, hidden_dim, num_layers, src_length, dropout=0.1, S = 24, num_experts = 16, device = "cpu", use_double=False
    ):
        # S : number of joints, default 24
        super(moe, self).__init__()
        self.model_type = "TransformerWithEncoderOnly"
        self.src_mask = None
        self.device = device
        self.pos_encoder = PositionalEncodingST(ninp, dropout, use_double=use_double)
        self.layers = nn.ModuleList([
            SpatialTemporalEncoderLayer(ninp, num_heads, num_experts, dropout, device=device, use_double=use_double)
            for _ in range(num_layers)
        ])

        # Use Linear instead of Embedding for continuous valued input
        self.encoder = nn.Linear(int(ntoken/S), ninp)

        # project to output
        self.project_1 = nn.Linear(ninp, int(ntoken/S))
        self.project_2 = nn.Linear(src_length, 1)
        self.ninp = ninp

        self.init_weights()

    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=1.):
        if max_len is None:
            max_len = tgt.shape[1]
        if self.training:
            max_len  = min(max_len, tgt.shape[1])# if training, we limit the upper bound
        B, T, E = src.shape # src: B, T, E
        data_chunk = torch.zeros(B, max_len, E).type_as(src.data)
        S = 24
        E = int(E/S)
        src_slice = src
        for i in range(max_len):
            src_slice_reshape = torch.reshape(src_slice, (B, T, S, E)) # B x T x S x E
            projected_src = self.encoder(src_slice_reshape) * np.sqrt(self.ninp) # B x T x S x E
            encoder_output = self.pos_encoder(projected_src)# B x T x S x E
            for layer in self.layers:
                encoder_output = layer(encoder_output)
            output = self.project_1(encoder_output) # B x T x S x E
            output = torch.reshape(src_slice_reshape, (B, T, S * E)) # B x T x (S x E)
            output = output.transpose(1, 2).contiguous() # B x (S x E) x T
            output = self.project_2(output)  # B x (S x E) x 1
            output = output.transpose(1, 2)  # B x 1 x (S x E)
            output = output + src_slice[:, -1:, :] # B x 1 x (S x E)
            data_chunk[:, i:i + 1, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            src_slice = src_slice[:,1:,:] # shift to the right
            if teacher_force:
                src_slice = torch.cat((src_slice, tgt[:,i:i+1,:]), axis=1)
            else:
                src_slice = torch.cat((src_slice, data_chunk[:,i:i+1,:]), axis=1)
        return data_chunk