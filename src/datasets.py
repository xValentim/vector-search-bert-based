import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

import pandas as pd
import arkad

import os
import json
import urllib.request
from pathlib import Path

class TextBertDataset(Dataset):
    def __init__(self, texts, labels, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )

        input_ids = tokens['input_ids'].squeeze(0)  # Remove the batch dimension
        attention_mask = tokens['attention_mask'].squeeze(0)

        return input_ids, attention_mask, label
    
def get_my_dataset(sample_size=None, sentence_length=128):
    # Load data
    paths =[x for x in os.listdir('../data') if x.endswith('.json')]

    dfs = []
    for path in paths:
        with open(f'../data/{path}', encoding='utf8') as f:
            data = json.load(f)[0]
            try:
                dfs.append(pd.DataFrame(data))
            except:
                data_ = []
                for idx , x in enumerate(data):
                    if x is None:
                        print(f"None in {idx}")
                    else:
                        data_.append(x)
                dfs.append(pd.DataFrame(data_))
    df = pd.concat(dfs).reset_index(drop=True)
    # Add quebra de linha '\n'
    all_data = df.copy()
    all_data['text'] = df['titulo'] + '\n\n' + df['subtitulo'] + '\n\n' + df['texto']
    all_data['class'] = df['label']
    all_data['url'] = df['url']
    all_data = all_data[['text', 'class', 'url']]
    label_counts = all_data['class'].value_counts()
    labels_above_100 = label_counts[label_counts > 400].index
    data = all_data[all_data['class'].isin(labels_above_100)]
    if sample_size is not None:
        data = data.sample(sample_size)
    output_data = data[['text', 'class']]
    
    label_encoder = {label: idx for idx, label in enumerate(output_data['class'].unique())}
    output_data['class'] = output_data['class'].map(label_encoder)
    X_train, X_test, y_train, y_test = train_test_split(output_data['text'], output_data['class'], test_size=0.2, random_state=42)
    classes = list(set(y_train))
    y_train_bin = torch.tensor([[classes.index(y) for y in y_train]]).T
    y_test_bin = torch.tensor([[classes.index(y) for y in y_test]]).T
    
    dataset_train = TextBertDataset(list(X_train), y_train_bin)
    dataset_test = TextBertDataset(list(X_test), y_test_bin)
    
    return dataset_train, dataset_test, label_encoder, classes
    
