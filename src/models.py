import torch
import torch.nn as nn
from transformers import BertModel
from arkad import BaseModel


class SamplingLayer(nn.Module):
    def __init__(self, encoding_size):
        super(SamplingLayer, self).__init__()
        self.encoding_size = encoding_size

    def forward(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        # print(std.shape)
        # print(mu.shape)
        codings = torch.normal(mu, std)
        return codings
    
class Encoder(BaseModel):
    def __init__(self,
                 hidden_dim=[768, 512, 256],
                 encoding_size=64,
                 dropout=0.4,
                 batch_norm_1d=True):
        super(Encoder, self).__init__()
        
        # ---- Embedding layer ----
        # self.n_special_tokens = n_special_tokens
        self.bert = BertModel.from_pretrained("bert-base-multilingual-uncased")
        self.encoding_size = encoding_size
        # -------------------------
        
        self.sampling_layer = SamplingLayer(encoding_size)
        if batch_norm_1d:
            self.batch_norm = nn.BatchNorm1d(768)
        else:
            self.batch_norm = nn.Identity()
        self.codings_mean = nn.Linear(hidden_dim[-1], encoding_size)
        self.codings_log_var = nn.Linear(hidden_dim[-1], encoding_size)
        
        self.dense_layers = []
        for i, dim in enumerate(hidden_dim):
            if i == 0:
                self.dense_layers.append(nn.Sequential(
                    nn.Linear(768, dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))
            else:
                self.dense_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim[i-1], dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))
        self.mlp = nn.Sequential(*self.dense_layers)
        
        for param in self.bert.parameters():
            param.requires_grad = False

    def _pool(self, x):
        x = torch.mean(x, dim=1)
        return x

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        x = self._pool(last_hidden_state)
        head_1 = x
        
        x = self.mlp(x)
        head_2 = x
        
        mu, log_var = self.codings_mean(x), self.codings_log_var(x)
        codings = self.sampling_layer(mu, log_var)

        return mu, log_var, codings, head_1, head_2
    
class Decoder(BaseModel):
    def __init__(self,
                 encoding_size=64,
                 hidden_dim=[256, 512, 768],
                 output_dim=768,
                 dropout=0.4):
        super(Decoder, self).__init__()
        self.encoding_size = encoding_size
        self.dense_layers = []
        for i, dim in enumerate(hidden_dim):
            if i == 0:
                self.dense_layers.append(nn.Sequential(
                    nn.Linear(encoding_size, dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))
            else:
                self.dense_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim[i-1], dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))
        self.mlp = nn.Sequential(*self.dense_layers)
        
    def forward(self, x):
        x = self.mlp(x)
        return x
    
class DecoderClassifier(BaseModel):
    def __init__(self,
                 encoding_size=64,
                 output_dim=1):
        super(DecoderClassifier, self).__init__()
        
        self.fc1 = nn.Linear(encoding_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
class VAE(BaseModel):
    def __init__(self,
                 hidden_dim=[768, 512, 256],
                 encoding_size=64,
                 output_dim_classifier=1,
                 dropout=0.4,
                 batch_norm_1d=False):
        super(VAE, self).__init__()
        self.encoder = Encoder(hidden_dim=hidden_dim,
                               encoding_size=encoding_size,
                               dropout=dropout,
                               batch_norm_1d=batch_norm_1d)
        self.decoder_classifier = DecoderClassifier(encoding_size=hidden_dim[-1],
                                                    output_dim=output_dim_classifier)
        self.decoder = Decoder(encoding_size,
                               hidden_dim=list(reversed(hidden_dim)),
                               output_dim=768,
                               dropout=dropout)
        
    def forward(self, input_ids, attention_mask):
        mu, log_var, codings, head_1, head_2 = self.encoder(input_ids, attention_mask)
        x = self.decoder(codings)
        y = self.decoder_classifier(head_2)
        return x, mu, log_var, head_1, head_2, y