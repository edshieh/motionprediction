# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn as nn
import os
from torch.nn import LayerNorm
from torch.nn import MultiheadAttention
from torch.nn.init import xavier_uniform_

from fairmotion.models import decoders
import random

# Set environment variable for MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def convert_on_device(tensor: torch.Tensor, device):
    return tensor.float() if device == "mips" else tensor.double()

class PositionalEncodingST(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device="cpu"):
        super(PositionalEncodingST, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            convert_on_device(torch.arange(0, d_model, 2), device) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) # T, E
        pe = pe.unsqueeze(0)# B, T, E
        pe = pe.unsqueeze(2)  # B, T, S, E
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: 32 x 120 x 24 x 128
        # print(x.is_contiguous(memory_format=torch.channels_last))
        x = x + self.pe[:,  x.size(1), :, :]
        # print(x.is_contiguous(memory_format=torch.channels_last))
        return self.dropout(x) # B, T, S, E

class SpatialTemporalEncoderLayer(nn.Module):
    def __init__(self, ninp, num_heads, hidden_dim, dropout, device):
        super(SpatialTemporalEncoderLayer, self).__init__()
        self.device = device
        self.SpatialMultiheadAttention = MultiheadAttention(ninp, num_heads, dropout)
        self.TemporalMultiheadAttention = MultiheadAttention(ninp, num_heads, dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.norm_1 = LayerNorm(ninp)
        self.norm_2 = LayerNorm(ninp)
        self.norm_3 = LayerNorm(ninp)

        self.linear_1 = nn.Linear(ninp, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, ninp)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            convert_on_device(mask, self.device)
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, x):
        B, T, S, E = x.shape # 32 x 120 x 24 x 128
        tx = torch.transpose(x, 0, 1) # 120 x 32 x 24 x 128
        tx = torch.reshape(tx, (T, B*S, E)) # 120 x (32*24) x 128
        # tx: input to temporal multihead (T, B*S, E)
        # t_mask :mask for temporal attention
        t_mask = self._generate_square_subsequent_mask(T).to(device=x.device,)
        tm, _ = self.TemporalMultiheadAttention(tx, tx, tx, attn_mask= t_mask) # 120 x (32*24) x 128
        # print(tm.is_contiguous(memory_format=torch.channels_last))
        tm = self.dropout_1(tm)
        # print(tm.is_contiguous(memory_format=torch.channels_last))
        tm = self.norm_1(tm + tx)
        # print(tm.is_contiguous(memory_format=torch.channels_last))
        tm = torch.transpose(tm, 0, 1) # (32*24) x 120 x 128
        # print(tm.is_contiguous(memory_format=torch.channels_last))
        tm = torch.reshape(tm, (B, S, T, E)) # 32 x 24 x 120 x 128
        # print(tm.is_contiguous(memory_format=torch.channels_last))
        tm = torch.transpose(tm, 1, 2) # 32 x 120 x 24 x 128
        # print(tm.is_contiguous(memory_format=torch.channels_last))

        sx = torch.reshape(x, (B * T, S, E)) # (32*120) x 24 x 128
        # print(sx.is_contiguous(memory_format=torch.channels_last))
        sx = torch.transpose(sx, 0, 1)# 24 x (32*120) x 128
        # print(sx.is_contiguous(memory_format=torch.channels_last))
        # sx: input to spatial multihead (S, B*T, E)
        sm, _ = self.SpatialMultiheadAttention(sx, sx, sx) # 24 x (32*120) x 128
        # print(sm.is_contiguous(memory_format=torch.channels_last))
        sm = self.dropout_2(sm)
        # print(sm.is_contiguous(memory_format=torch.channels_last))
        sm = self.norm_2(sm + sx)
        # print(sm.is_contiguous(memory_format=torch.channels_last))
        sm = torch.transpose(sm, 0, 1) # (32*120) x 24 x 128
        # print(sm.is_contiguous(memory_format=torch.channels_last))
        sm = torch.reshape(sm, (B, T, S, E)) # 32 x 120 x 24 x 128
        # print(sm.is_contiguous(memory_format=torch.channels_last))

        # add temporal + spatial
        input = tm + sm
        # feed forward
        # B, T, S, hidden: 32, 120, 24, hidden
        xx = self.linear_1(input)
        # print(xx.is_contiguous(memory_format=torch.channels_last))
        xx = nn.ReLU()(xx)
        # print(xx.is_contiguous(memory_format=torch.channels_last))
        xx = self.linear_2(xx)
        # print(xx.is_contiguous(memory_format=torch.channels_last))
        # add and norm
        xx = self.dropout_3(xx)
        # print(xx.is_contiguous(memory_format=torch.channels_last))
        # print(self.norm_3(xx + input).is_contiguous(memory_format=torch.channels_last))
        return self.norm_3(xx + input)


class TransformerSpatialTemporalModel(nn.Module):
    def __init__(
        self, ntoken, ninp, num_heads, hidden_dim, num_layers, src_length, dropout=0.1, S = 24, device="cpu"
    ):
        # S : number of joints, default 24
        super(TransformerSpatialTemporalModel, self).__init__()
        self.model_type = "TransformerWithEncoderOnly"
        self.src_mask = None
        self.device = device
        self.pos_encoder = PositionalEncodingST(ninp, dropout)
        self.layers = nn.ModuleList([
            SpatialTemporalEncoderLayer(ninp, num_heads, hidden_dim, dropout, device)
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
        B, T, E = src.shape # src: B, T, E, 32 x 120 x 216
        data_chunk = torch.zeros(B, max_len, E).type_as(src.data)
        S = 24
        E = int(E/S)
        src_slice = src
        for i in range(max_len):
            src_slice_reshape = torch.reshape(src_slice, (B, T, S, E)) # 32 x 120 x 24 x 9
            # print("1.",src_slice_reshape.is_contiguous(memory_format=torch.channels_last))
            projected_src = self.encoder(src_slice_reshape) * np.sqrt(self.ninp) # 32 x 120 x 24 x 128
            # print("2.",projected_src.is_contiguous(memory_format=torch.channels_last))
            encoder_output = self.pos_encoder(projected_src)# 32 x 120 x 24 x 128
            # print("3.",encoder_output.is_contiguous(memory_format=torch.channels_last))
            # print("encoder_output:", encoder_output.shape)
            # print(len(self.layers))
            for layer in self.layers:
                encoder_output = layer(encoder_output)
                # print("encoder_output:", encoder_output.shape)
                # print("4.",encoder_output.is_contiguous(memory_format=torch.channels_last))
            # print("Shape before project_1:", encoder_output.shape)
            # print("5.",encoder_output.is_contiguous(memory_format=torch.channels_last))
            output = self.project_1(encoder_output) # 32 x 120 x 24 x 9
            # print("Shape after project_1:", output.shape)
            # print("6.",src_slice_reshape.is_contiguous(memory_format=torch.channels_last))
            output = torch.reshape(src_slice_reshape, (B, T, S * E)) # 32 x 120 x 216
            # print("7.",output.is_contiguous(memory_format=torch.channels_last))
            output = output.transpose(1, 2).contiguous() # 32 x 216 x 120
            # print("8.",output.is_contiguous(memory_format=torch.channels_last))
            output = self.project_2(output)  # 32 x 216 x 1
            # print(output.is_contiguous(memory_format=torch.channels_last))
            output = output.transpose(1, 2)  # 32 x 1 x 216
            # print(output.is_contiguous(memory_format=torch.channels_last))
            output = output + src_slice[:, -1:, :] # 32 x 1 x 216
            # print(output.is_contiguous(memory_format=torch.channels_last))
            data_chunk[:, i:i + 1, :] = output
            # print(data_chunk.is_contiguous(memory_format=torch.channels_last))
            teacher_force = random.random() < teacher_forcing_ratio
            src_slice = src_slice[:,1:,:] # shift to the right
            # print(src_slice.is_contiguous(memory_format=torch.channels_last))
            if teacher_force:
                src_slice = torch.cat((src_slice, tgt[:,i:i+1,:]), axis=1)
            else:
                src_slice = torch.cat((src_slice, data_chunk[:,i:i+1,:]), axis=1)
            # print(src_slice.is_contiguous(memory_format=torch.channels_last))
            # print(data_chunk.is_contiguous(memory_format=torch.channels_last))
        return data_chunk