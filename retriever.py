import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel as PydanticBaseModel
from src.models import VAE

class Output(PydanticBaseModel):
    content: str
    url: str
    score: float

class Retriever:
    def __init__(self, 
                 path_saved_model='./models/vae_model_state_dict_2.pth', 
                 sample_size=None, 
                 path_saved_index='./data/mu_outputs.csv'):
        
        # Defina os hiperparâmetros
        hidden_dim = [768, 512, 256]
        encoding_size = 128
        learning_rate = 3e-4
        num_epochs = 15
        batch_size = 64
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Recrie a instância do modelo (use os mesmos hiperparâmetros)
        model_loaded = VAE(
            hidden_dim=hidden_dim,
            encoding_size=encoding_size,
            output_dim_classifier=7,
            dropout=0.5,
            batch_norm_1d=True
        )

        model_loaded.load_state_dict(torch.load(path_saved_model))
        model_loaded.to('cpu')
        
        self.model = model_loaded
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        
        self.data = pd.read_csv('./data/data.csv')
        if sample_size:
            self.data = self.data.sample(sample_size)
        
        if path_saved_index is None:
            
            self.tokenized_data = tokenizer(self.data['text'].tolist(), padding=True, truncation=True, return_tensors='pt')
            self.dataset = TensorDataset(self.tokenized_data['input_ids'], self.tokenized_data['attention_mask'])
            self.dataloader = DataLoader(self.dataset, batch_size=128)  # Ajuste o batch_size conforme a capacidade da GPU
            
            mu_outputs = []
            model_loaded.eval()  # Colocar o modelo em modo de avaliação
            model_loaded.to(device)  # Mova o modelo para a cuda

            with torch.no_grad():  # Desativar o cálculo de gradiente para economizar memória
                for batch in tqdm(self.dataloader):
                    input_ids, attention_mask = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    outputs = model_loaded.encoder(input_ids=input_ids, attention_mask=attention_mask)
                    mu_outputs.append(outputs[0])

            # Concatenar os batches para ter o resultado final
            self.mu_outputs = torch.cat(mu_outputs, dim=0).cpu().detach().numpy()
        else:
            self.mu_outputs = pd.read_csv(path_saved_index)
            self.mu_outputs = self.mu_outputs.values
        
    def invoke(self, query, k=10):
        self.model.eval()
        query_processed = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True)
        query_embedding_mu = self.model.encoder(input_ids=query_processed['input_ids'].to('cpu'), 
                                                  attention_mask=query_processed['attention_mask'].to('cpu'))[0]
        query_embedding_mu = query_embedding_mu.cpu().detach().numpy()[0]
        query_embedding_mu = query_embedding_mu.reshape(1, -1)
        
        scores = cosine_similarity(self.mu_outputs, query_embedding_mu)
        idxs = np.argsort(scores, axis=0)[::-1][:k]
        idxs_and_scores = np.array([[idx[0], scores[idx[0]][0]] for idx in idxs])
        
        output_content = self.data.iloc[idxs_and_scores[:, 0]]['text'].values
        output_url = self.data.iloc[idxs_and_scores[:, 0]]['url'].values
        output_scores = idxs_and_scores[:, 1]
        
        return output_content, output_url, output_scores
    
    def query(self, query, k=3):
        output_content, output_url, output_scores = self.invoke(query, k)
        output = [Output(content=output_content[i],
                         url=output_url[i],
                         score=output_scores[i]
                    ) for i in range(k)]
        return output