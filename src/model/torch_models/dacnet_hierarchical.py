import torch
import torch.nn as nn

from .layers import ConvBlock, FCBlock, LSTMBlock, AttnBlock
from .itransformer import iTransformer
from typing import Callable

class LinearScorer(nn.Module):
    def __init__(self, emb_len=128, n_heads=2, binarize_scores=True):
        super(LinearScorer, self).__init__()
        self.binarize_scores = binarize_scores

        self.proj = nn.Linear(emb_len, n_heads) 
        self.score_function = nn.Softmax(dim=1)

    def forward(self, inputs):
        # inputs -> [B, E, S]
        scores =  self.score_function(self.proj(inputs.flatten(1))) # Output -> [B, Heads]

        if self.binarize_scores:
            scores = scores + (torch.round(scores)-scores).detach()

        return scores
    
class AttnScorer(nn.Module):
    def __init__(self, emb_len=128, n_heads=2, binarize_scores=True):
        super(AttnScorer, self).__init__()
        self.binarize_scores = binarize_scores

        self.proj = nn.MultiheadAttention(emb_len, 1) 
        self.keyvalue = nn.Parameter(torch.rand(n_heads, emb_len))

    def forward(self, inputs):
        # inputs -> [B, E, S]
        scores =  self.proj(inputs.flatten(1), self.keyvalue, self.keyvalue)[1] # Output -> [B, Heads]

        if self.binarize_scores:
            scores = scores + (torch.round(scores)-scores).detach()

        return scores

class Head(nn.Module):
    def __init__(self, emb_in, emb_out=1, emb_len=8, seq_len=8, pred_len=8, kernel_size=3, dropout=0.1, basic_block='cnn'):
        super(Head, self).__init__()
        self.basic_block = basic_block
        self.emb_in = emb_in
        self.pred_len = pred_len
        self.emb_out = emb_out
        
        if basic_block == 'cnn':
            self.layer = ConvBlock(emb_in, emb_len, kernel_size, dropout)
            self.out_conv = nn.Conv1d(seq_len, pred_len, kernel_size=1)
            self.out_layer = nn.Linear(emb_len, emb_out)
        elif basic_block == 'fc':
            self.layer = FCBlock(emb_in, emb_len, dropout)
            self.out_layer = nn.Linear(emb_len, emb_out*pred_len)
        elif basic_block == 'lstm':
            self.layer = LSTMBlock(emb_in, emb_len, dropout)
            self.out_layer = nn.Linear(emb_len*seq_len, emb_out*pred_len) 
        elif basic_block == 'attn':
            self.layer = AttnBlock(emb_in, dropout)
            self.out_layer = nn.Linear(emb_in*seq_len, emb_out*pred_len)
        elif basic_block == 'itransformer':
            self.layer = iTransformer(pred_len=pred_len, seq_len=seq_len, dropout=dropout, d_model=emb_len)
            self.out_layer = nn.Linear(emb_in, emb_out*pred_len)
    
    def forward(self, inputs):
        # inputs -> [B, E, S]

        x = inputs
        out = self.layer(x)
        
        if self.basic_block == 'cnn':
            out = self.out_conv(out)
            out = self.out_layer(out)
        elif self.basic_block == 'lstm':
            out = self.out_layer(out[0].flatten(1)).reshape(-1, self.pred_len, self.emb_out)
        else:
            out = self.out_layer(out.flatten(1)).reshape(-1, self.pred_len, self.emb_out)

        return out # out -> [B, P, 1]

class BTreeNN(nn.Module):
    def __init__(self, batch_size=32, emb_in=1, emb_out=1, seq_len=8, pred_len=8, emb_len=128, n_layers=2, kernel_size=None, dropout=0.1, layer_index=0, body_block='cnn', head_block='fc', binarize_scores=True, scorer="linear", detach_parent=False, device='cpu'):
        super(BTreeNN, self).__init__()
        self.device  = device
        self.emb_in = emb_in
        self.emb_out = emb_out
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.layer_index = layer_index
        self.n_layers = n_layers
        self.body_block = body_block
        self.head_block = head_block
        self.detach_parent = detach_parent

        self.conv = nn.Conv1d(emb_in, emb_len, 3)
        self.relu = nn.ReLU()

        if self.body_block == 'cnn':
            self.principal_block = ConvBlock(emb_in, emb_len, 3, dropout)
        elif self.body_block == 'fc':
            emb_in = emb_in*self.seq_len if layer_index==0 else emb_in
            self.principal_block = FCBlock(emb_in, emb_len, dropout)
        elif self.body_block == 'lstm':
            self.principal_block = LSTMBlock(emb_in, emb_len, dropout)
        elif self.body_block == 'attn':
            self.principal_block = AttnBlock(emb_len, dropout)
            	
        input_size = emb_in+emb_len
        scorer_size = input_size

        if self.body_block != 'fc':
            scorer_size = scorer_size*seq_len

        if scorer == "linear":
            self.scorer = LinearScorer(emb_len=scorer_size, n_heads=2, binarize_scores=binarize_scores)
        else:
            self.scorer = AttnScorer( emb_len=scorer_size, n_heads=2, binarize_scores=binarize_scores)

        if layer_index == (n_layers-1):
            self.head_1 = Head(emb_in = input_size, emb_out=emb_out, emb_len=emb_len, seq_len=seq_len, pred_len=pred_len, 
                                                        kernel_size=kernel_size, dropout=dropout, basic_block=head_block)
            self.head_2 = Head(emb_in=input_size, emb_out=emb_out, emb_len=emb_len, seq_len=seq_len, pred_len=pred_len, 
                                                        kernel_size=kernel_size, dropout=dropout, basic_block=head_block)
        else:
            self.split_head_1 = BTreeNN(emb_in=input_size, emb_len=emb_len, seq_len=seq_len, pred_len=pred_len, 
                                                        kernel_size=kernel_size, dropout=dropout, layer_index=layer_index+1, body_block=body_block, head_block=head_block, binarize_scores=binarize_scores, scorer=scorer, detach_parent=detach_parent)
            self.split_head_2 = BTreeNN(emb_in=input_size, emb_len=emb_len, seq_len=seq_len, pred_len=pred_len, 
                                                        kernel_size=kernel_size, dropout=dropout, layer_index=layer_index+1, body_block=body_block, head_block=head_block, binarize_scores=binarize_scores, scorer=scorer, detach_parent=detach_parent)
    
    def forward(self, inputs):

        batch_size = inputs.shape[0]
            
        self.scores = torch.zeros((batch_size, 2),  device=inputs.device).double() # , device=self.device
        self.outputs = torch.zeros((2, batch_size, self.pred_len, self.emb_out), device=inputs.device).double() #, device=self.device

        inputs_embedded = self.principal_block(inputs)

        if self.body_block == 'lstm':
            inputs_embedded = inputs_embedded[0]
        
        if self.detach_parent:
            inputs_embedded = inputs_embedded.detach()
        
        inputs_embedded = torch.cat((inputs, inputs_embedded), dim=-1)

        self.scores = self.scorer(inputs_embedded)

        if self.layer_index < (self.n_layers-1):

            self.outputs[0] = self.split_head_1(inputs_embedded)
            self.outputs[1] = self.split_head_2(inputs_embedded)

        else:
            self.outputs[0] = self.head_1(inputs_embedded)
            self.outputs[1] = self.head_2(inputs_embedded)

        scores = self.scores.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
 
        outputs_scored = self.outputs * scores
        
        output = torch.sum(outputs_scored, dim=0)

        return output
        

class DACNetHierarchical(nn.Module):
    """
    Mejoras:
        * Incluir varias capas a Head
        * Incluir centroides de scorer
    
    """

    def __init__(self, parameters: dict, n_features_in: int = 1, n_features_out: int = 1, device: torch.device = None) -> None:
        super(DACNetHierarchical, self).__init__()
        self.emb_in = n_features_in

        self.pred_len = parameters['dataset']['params']['pred_len']
        self.seq_len = parameters['dataset']['params']['seq_len']
        self.batch_size = parameters['model']['params']['batch_size']
        self.emb_len = parameters['model']['params']['units']
        self.binarize_scores = parameters['model']['params']['binarize_scores']
        self.body_block = parameters['model']['params']['body_block']
        self.head_block = parameters['model']['params'].get('head_block', self.body_block)
        self.dropout = parameters['model']['params']['dropout']
        self.n_layers = parameters['model']['params']['layers']
        self.scorer = parameters['model']['params']['scorer']
        self.detach_parent = parameters['model']['params']['detach_parent']

        if device==None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.split_head = BTreeNN(emb_in=n_features_in, emb_out=n_features_out, emb_len=self.emb_len, seq_len=self.seq_len, pred_len=self.pred_len, n_layers=self.n_layers, 
                                                        kernel_size=3, dropout=self.dropout, layer_index=0, body_block=self.body_block, head_block=self.head_block, 
                                                        scorer=self.scorer, detach_parent=self.detach_parent, binarize_scores=self.binarize_scores, device=device)
        
        self.double()
        
    def compute_loss(self, true: torch.Tensor, pred: torch.Tensor, criterion: Callable) -> torch.Tensor:
        return criterion(true, pred)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs -> [B, S, E_IN]
        if self.body_block == 'fc': 
            inputs = inputs.flatten(1)

        output = self.split_head(inputs)

        return output

