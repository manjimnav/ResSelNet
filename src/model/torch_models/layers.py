import torch.nn as nn

class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, padding_mode='replicate', **kwargs)
   
    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, :-self.conv.padding[0]]  # remove trailing padding
        return x 

class ConvBlock(nn.Module):

    def __init__(self,emb_len, d_model=512, kernel_size=3,  dropout=0.1):
        super(ConvBlock, self).__init__()
        self.conv1 = CausalConv1d(in_channels=emb_len, out_channels=d_model, kernel_size=kernel_size)
        self.activation1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.dropout(x)

        return x.transpose(1, 2)

class FCBlock(nn.Module):

    def __init__(self, emb_len, d_model=512,  dropout=0.1):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_features=emb_len, out_features=d_model)
        #self.activation1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        #x = self.activation1(x)

        x = self.dropout(x)

        return x

class LSTMBlock(nn.Module):

    def __init__(self, emb_len, d_model=512,  dropout=0.1):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=emb_len, hidden_size=d_model,batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x):
        if type(x) == tuple and x[0].shape[2] == self.d_model:
            x = self.lstm(x[0], x[1])
        elif type(x) == tuple:
            x = self.lstm(x[0])
        else:
             x = self.lstm(x)

        return x

class AttnBlock(nn.Module):

    def __init__(self, emb_in, dropout=0.1):
        super(AttnBlock, self).__init__()
        self.attn = nn.MultiheadAttention(emb_in, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        return self.dropout(self.attn(x, x, x)[0])