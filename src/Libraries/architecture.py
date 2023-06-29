import torch
from torch import nn
import torch.nn.functional as F
import math


class Cnn(nn.Module):
    def __init__(self, num_of_classes):
        super(Cnn, self).__init__()
        self.scat_conv = nn.Sequential(
            nn.BatchNorm2d(1), nn.Conv2d(1, 10, 5), nn.MaxPool2d(2), nn.ReLU(),
            nn.BatchNorm2d(10), nn.Conv2d(10, 20, 3), nn.MaxPool2d(2), nn.Dropout2d(), nn.ReLU()
        )
        self.spec_conv = nn.Sequential(
            nn.BatchNorm2d(1), nn.Conv2d(1, 50, (1, 5)), nn.ReLU(),
            nn.BatchNorm2d(50), nn.Conv2d(50, 100, 3), nn.Dropout2d(), nn.ReLU()
        )
        self.liti_conv = nn.Sequential(
            nn.BatchNorm2d(1), nn.Conv2d(1, 70, (1, 7)), nn.ReLU(),
            nn.BatchNorm2d(70), nn.Conv2d(70, 140, (1, 5)), nn.ReLU(),
            nn.BatchNorm2d(140), nn.Conv2d(140, 200, 3), nn.Dropout2d(), nn.ReLU()
        )
        self.scat_fc = nn.Sequential(nn.Linear(1680, 50), nn.ReLU(), nn.Dropout2d())
        self.spec_fc = nn.Sequential(nn.Linear(5200, 50), nn.ReLU(), nn.Dropout2d())
        self.liti_fc = nn.Sequential(nn.Linear(4800, 50), nn.ReLU(), nn.Dropout2d())
        self.li_w_fc = nn.Sequential(nn.BatchNorm1d(4), nn.ReLU(), nn.Dropout2d())
        self.size_fc = nn.Sequential(nn.BatchNorm1d(1), nn.ReLU(), nn.Dropout2d())
        self.final_fc = nn.Sequential(nn.Linear(155, num_of_classes), nn.LogSoftmax(dim=1))

    def forward(self, data):
        x1 = self.scat_conv(data['scattering image'])
        x2 = self.spec_conv(data['spectrum'])
        x3 = self.liti_conv(data['lifetime'])

        x1 = x1.view(-1, 1680)
        x2 = x2.view(-1, 5200)
        x3 = x3.view(-1, 4800)

        x1 = self.scat_fc(x1)
        x2 = self.spec_fc(x2)
        x3 = self.liti_fc(x3)
        x4 = self.li_w_fc(data['lifetime weights'])
        x5 = self.size_fc(data['size'])

        x = torch.cat((x1, x2, x3, x4, x5), 1)
        return self.final_fc(x)



class TransformerEncoder(nn.Module):
    def __init__(self, fc_dim, num_layers, num_heads, dropout, num_classes):
        super().__init__()
        self.fc_dim = fc_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.final_fc = nn.Linear(150, num_classes)

        embed_dim_sc = 20
        self.pos_enc_sc = PositionalEncoding(embed_dim_sc, self.dropout)
        encoder_layers_sc = nn.TransformerEncoderLayer(embed_dim_sc, self.num_heads, dim_feedforward=fc_dim, dropout=self.dropout)
        self.transformer_encoder_sc = nn.TransformerEncoder(encoder_layers_sc, num_layers=self.num_layers)
        self.fc_sc = nn.Linear(2400, 50)

        embed_dim_sp = 32
        self.pos_enc_sp = PositionalEncoding(embed_dim_sp, self.dropout)
        encoder_layers_sp = nn.TransformerEncoderLayer(embed_dim_sp, self.num_heads, dim_feedforward=fc_dim, dropout=self.dropout)
        self.transformer_encoder_sp = nn.TransformerEncoder(encoder_layers_sp, num_layers=self.num_layers)
        self.fc_sp = nn.Linear(128, 50)

        embed_dim_li = 4
        self.pos_enc_li = PositionalEncoding(embed_dim_li, self.dropout)
        encoder_layers_li = nn.TransformerEncoderLayer(embed_dim_li, self.num_heads, dim_feedforward=fc_dim, dropout=self.dropout)
        self.transformer_encoder_li = nn.TransformerEncoder(encoder_layers_li, num_layers=self.num_layers)
        self.fc_li = nn.Linear(96, 50)


    def forward(self, data):
        x1 = self.pos_enc_sc(data['scattering image'])
        x1 = self.transformer_encoder_sc(x1)
        x1 = torch.flatten(torch.transpose(x1, 0, 1), 1, 2)
        x1 = self.fc_sc(x1)
        
        x2 = self.pos_enc_sp(data['spectrum'])
        x2 = self.transformer_encoder_sp(x2)
        x2 = torch.flatten(torch.transpose(x2, 0, 1), 1, 2)
        x2 = self.fc_sp(x2)
        
        x3 = self.pos_enc_li(data['lifetime'])
        x3 = self.transformer_encoder_li(x3)
        x3 = torch.flatten(torch.transpose(x3, 0, 1), 1, 2)
        x3 = self.fc_li(x3)

        x = torch.cat((x1, x2, x3), 1)
        x = self.final_fc(x)
        return F.log_softmax(x, dim=1)
   

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x): #x: Tensor, shape [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class CnnTL(nn.Module):
    def __init__(self, num_of_classes):
        super(CnnTL, self).__init__()
        self.scat_conv = nn.Sequential(
            nn.BatchNorm2d(1), nn.Conv2d(1, 10, 5), nn.MaxPool2d(2), nn.ReLU(),
            nn.BatchNorm2d(10), nn.Conv2d(10, 20, 3), nn.MaxPool2d(2), nn.Dropout2d(), nn.ReLU()
        )
        self.spec_conv = nn.Sequential(
            nn.BatchNorm2d(1), nn.Conv2d(1, 50, (1, 5)), nn.ReLU(),
            nn.BatchNorm2d(50), nn.Conv2d(50, 100, 3), nn.Dropout2d(), nn.ReLU()
        )
        self.liti_conv = nn.Sequential(
            nn.BatchNorm2d(1), nn.Conv2d(1, 70, (1, 7)), nn.ReLU(),
            nn.BatchNorm2d(70), nn.Conv2d(70, 140, (1, 5)), nn.ReLU(),
            nn.BatchNorm2d(140), nn.Conv2d(140, 200, 3), nn.Dropout2d(), nn.ReLU()
        )
        self.scat_fc = nn.Sequential(nn.Linear(1680, 50), nn.ReLU(), nn.Dropout2d())
        self.spec_fc = nn.Sequential(nn.Linear(5200, 50), nn.ReLU(), nn.Dropout2d())
        self.liti_fc = nn.Sequential(nn.Linear(4800, 50), nn.ReLU(), nn.Dropout2d())
        self.li_w_fc = nn.Sequential(nn.BatchNorm1d(4), nn.ReLU(), nn.Dropout2d())
        self.size_fc = nn.Sequential(nn.BatchNorm1d(1), nn.ReLU(), nn.Dropout2d())
        self.final_fc = nn.Sequential(nn.Linear(155, num_of_classes), nn.LogSoftmax(dim=1))
        
    def forward(self, data, important_classes):
        y = torch.zeros((len(data), len(important_classes)))
        softmax = nn.Softmax(dim = 1)

        for i, sample in enumerate(data):
            x1 = self.scat_conv(sample['scattering image'])
            x2 = self.spec_conv(sample['spectrum'])
            x3 = self.liti_conv(sample['lifetime'])

            x1 = x1.view(-1, 1680)
            x2 = x2.view(-1, 5200)
            x3 = x3.view(-1, 4800)

            x1 = self.scat_fc(x1)
            x2 = self.spec_fc(x2)
            x3 = self.liti_fc(x3)
            x4 = self.li_w_fc(sample['lifetime weights'])
            x5 = self.size_fc(sample['size'])

            x = torch.cat((x1, x2, x3, x4, x5), 1)

            x = self.final_fc(x)
            x = softmax(x)
            x = torch.sum(x, 0)
            x = x[important_classes]
            y[i, :] = x

        y = y - torch.min(y, 0)[0]
        y = y / torch.max(y, 0)[0]
        return y
    


def load_model_weights(model, path):
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    

def move_to_cuda(thing, gpu):
    if torch.cuda.is_available():
        thing.cuda(gpu)
    else:
        print("Cuda is not available.")