import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Encoder, EncoderLayer, FullAttention, AttentionLayer, DataEmbedding_inverted

import numpy as np

class iTransformer(nn.Module):


    def __init__(self, parameters=None, pred_len=None, seq_len=None, d_model=None, output_attention=False, dropout=None, n_layers=1, n_heads=2, use_norm=True, n_features_in=None, n_features_out=None):
        super(iTransformer, self).__init__()


        if parameters != None:
            self.pred_len = parameters['dataset']['params']['pred_len']
            self.seq_len = parameters['dataset']['params']['seq_len']
            self.d_model = parameters['model']['params']['units']
            self.output_attention = parameters['model']['params']['output_attention']
            self.dropout = parameters['model']['params']['dropout']
            self.n_layers = parameters['model']['params']['layers']
            self.n_heads = parameters['model']['params']['n_heads']
            self.use_norm = parameters['model']['params']['use_norm']
        else:
            self.pred_len = pred_len
            self.seq_len = seq_len
            self.d_model = d_model
            self.output_attention = output_attention
            self.dropout = dropout
            self.n_layers = n_layers
            self.n_heads = n_heads
            self.use_norm = use_norm

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, self.dropout)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    dropout=self.dropout
                    ) for l in range(self.n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.projector = nn.Linear(self.d_model, self.pred_len, bias=True)

        self.double()

    def forward(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, None) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, :]
