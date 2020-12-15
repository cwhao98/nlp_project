import torch
import torch.nn as nn
import torch.nn.functional as F

from base import Base


class GlobalMaxPool1d(nn.Module):
    
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    
    def forward(self, x):
        ''' x: shape(batch, channel, seq_len)'''
        return F.max_pool1d(x, kernel_size=x.shape[-1])

class TextCNN(Base):

    def __init__(self,
                 args,
                 kernel_sizes=[3, 4, 5],
                 num_channels=[50, 50, 50],
                 num_emotion=2):

        super(TextCNN, self).__init__(args=args)

        self.embedding = nn.Embedding(args.vocab_size, args.emb_size)
        # self.constant_embedding = nn.Embedding(vocab_size, emb_size) # frozen
        self.pool = GlobalMaxPool1d()
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        self.encoder = nn.ModuleList()
        for k, c in zip(kernel_sizes, num_channels):
            self.encoder.append(nn.Conv1d(in_channels=args.emb_size * 1, out_channels=c, kernel_size=k))
        
        self.decoder = nn.Linear(sum(num_channels), args.num_emotion)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        '''inputs: shape(batch, seq_len) 
        '''
        # emb = torch.cat([self.embedding(inputs), self.constant_embedding(inputs)], dim=2) # (batch, seq_len, emb_size)
        # print(inputs)
        emb = self.embedding(inputs)
        emb = emb.permute(0, 2, 1) # (batch, emb_size, seq_len)
        encoding = torch.cat([self.pool(self.relu(conv(emb))).squeeze(-1) for conv in self.encoder], dim=1)
        output = self.decoder(self.drop(encoding)) # (batch, num_emotion)
        # print(emb)
        return output










