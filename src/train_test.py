import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model,
             dataloader,
             criterions=[],
             metrics=['accuracy', 'precision', 'recall', 'f1'], 
             device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    model.eval()
    total_loss = 0
    total_kl_loss = 0
    total_recon_loss = 0
    total_classification_loss = 0
    
    beta = 1.2
    classification_weight = 2.0
    reconstruction_weight = 1.0
    
    all_preds = []
    all_labels = []
    
    reconstruction_loss_fn, classification_loss_fn = criterions
    
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(dataloader)):
            
            input_ids, attention_mask, y = data
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y = y.to(device)
            # Adjust y to have the correct shape and data type
            y = y.squeeze(-1)  # Remove the extra dimension
            y = y.long()       # Ensure y is of type torch.LongTensor
            
            reconstructed_x, mu, log_var, head_1, head_2, y_hat = model(input_ids, attention_mask)

            # Reconstrução entre 'head' e 'reconstructed_x'
            recon_loss = reconstruction_loss_fn(reconstructed_x, head_1)

            # Compute o termo de divergência KL
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # Normalizar o KL loss pelo tamanho do batch
            kl_loss /= input_ids.shape[0]
            
            # Classificação
            classification_loss = classification_loss_fn(y_hat, y)

            # preds
            preds = torch.argmax(y_hat, dim=1)
            all_preds.extend(preds)
            all_labels.extend(y)

            # Perda total
            loss = (reconstruction_weight * recon_loss) + (beta * kl_loss) + (classification_weight * classification_loss)

            total_loss += loss.item()
            total_kl_loss += beta * kl_loss.item()
            total_recon_loss += reconstruction_weight * recon_loss.item()
            total_classification_loss += classification_weight * classification_loss.item()
    
    all_preds = [p.item() for p in all_preds]
    all_labels = [l.item() for l in all_labels]
    
    accuracy = accuracy_score(all_labels, all_preds) if 'accuracy' in metrics else 0.0
    precision = precision_score(all_labels, all_preds, zero_division=0) if 'precision' in metrics else 0.0
    recall = recall_score(all_labels, all_preds, zero_division=0) if 'recall' in metrics else 0.0
    f1 = f1_score(all_labels, all_preds, zero_division=0) if 'f1' in metrics else 0.0
    
    avg_loss = total_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_classification_loss = total_classification_loss / len(dataloader)
    
    model.train()
    
    return avg_loss, avg_kl_loss, avg_recon_loss, avg_classification_loss, accuracy, precision, recall, f1

# Função de treinamento
def train(model,
          dataset,
          optimizer,
          num_epochs,
          batch_size,
          criterions=[],
          validation_split=0.2,
          random_seed=42,
          metrics=['accuracy'],
          device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.train()
    beta = 1.2
    classification_weight = 2.0
    reconstruction_weight = 1.0
    history_train = {
        'total_loss_train': [],
        'reconstruction_loss_train': [],
        'kl_loss_train': [],
        'classification_loss_train': []
    }
    
    history_val = {
        'total_loss_val': [],
        'reconstruction_loss_val': [],
        'kl_loss_val': [],
        'classification_loss_val': []
    }
    
    history_metrics_train = {
        'accuracy_train': [],
        'precision_train': [],
        'recall_train': [],
        'f1_train': []
    }
    
    history_metrics_val = {
        'accuracy_val': [],
        'precision_val': [],
        'recall_val': [],
        'f1_val': []
    }
    
    # Determinar o tamanho dos conjuntos de treinamento e validação
    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    
    
    # Dividir o dataset
    if validation_split > 0.0:
        generator = torch.Generator().manual_seed(random_seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        print(f"Dataset dividido em {train_size} amostras para treinamento e {val_size} para validação.")
    else:
        train_dataset = dataset
        val_dataset = None
        print("Nenhuma divisão de validação foi realizada. Usando todo o dataset para treinamento.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    
    reconstruction_loss_fn, classification_loss_fn = criterions
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_kl_loss = 0
        total_recon_loss = 0
        total_classification_loss = 0
        all_preds_train = []
        all_labels_train = []
        
        
        for batch_idx, data in tqdm(enumerate(train_loader)):
            
            input_ids, attention_mask, y = data
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y = y.to(device)
            # Adjust y to have the correct shape and data type
            y = y.squeeze(-1)  # Remove the extra dimension
            y = y.long()       # Ensure y is of type torch.LongTensor
            
            optimizer.zero_grad()

            reconstructed_x, mu, log_var, head_1, head_2, y_hat = model(input_ids, attention_mask)

            # Reconstrução entre 'head' e 'reconstructed_x'
            recon_loss = reconstruction_loss_fn(reconstructed_x, head_1)

            # Compute o termo de divergência KL
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # Normalizar o KL loss pelo tamanho do batch
            kl_loss /= input_ids.shape[0]
            
            # classes
            # print(y_hat.shape)
            # print(y.shape)
            # print(y_hat)
            # print(y)
            
            # Classificação
            classification_loss = classification_loss_fn(y_hat, y)
            
            # Preds
            preds = torch.argmax(y_hat, dim=1)
            labels = y
            all_preds_train.extend(preds)
            all_labels_train.extend(labels)

            # Perda total
            loss = (reconstruction_weight * recon_loss) + (beta * kl_loss) + (classification_weight * classification_loss)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_kl_loss += beta * kl_loss.item()
            total_recon_loss += reconstruction_weight * recon_loss.item()
            total_classification_loss += classification_weight * classification_loss.item()
            
        if val_loader:
            val_loss, val_kl_loss, val_recon_loss, val_classification_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_loader, metrics=['accuracy'], criterions=criterions, device=device)
            history_val['total_loss_val'].append(val_loss)
            history_val['kl_loss_val'].append(val_kl_loss)
            history_val['reconstruction_loss_val'].append(val_recon_loss)
            history_val['classification_loss_val'].append(val_classification_loss)
            
            history_metrics_val['accuracy_val'].append(val_accuracy)
            history_metrics_val['precision_val'].append(val_precision)
            history_metrics_val['recall_val'].append(val_recall)
            history_metrics_val['f1_val'].append(val_f1)

        avg_loss = total_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_classification_loss = total_classification_loss / len(train_loader)
        
        all_preds_train = [p.item() for p in all_preds_train]
        all_labels_train = [l.item() for l in all_labels_train]
        
        accuracy_train = accuracy_score(all_labels_train, all_preds_train) if 'accuracy' in metrics else 0.0
        precision_train = precision_score(all_labels_train, all_preds_train, zero_division=0) if 'precision' in metrics else 0.0
        recall_train = recall_score(all_labels_train, all_preds_train, zero_division=0) if 'recall' in metrics else 0.0
        f1_train = f1_score(all_labels_train, all_preds_train, zero_division=0) if 'f1' in metrics else 0.0
        
        history_train['total_loss_train'].append(avg_loss)
        history_train['kl_loss_train'].append(avg_kl_loss)
        history_train['reconstruction_loss_train'].append(avg_recon_loss)
        history_train['classification_loss_train'].append(avg_classification_loss)
        
        if val_loader:
        
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Loss: {avg_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, Classification Loss: {avg_classification_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}, Validation KL Loss: {val_kl_loss:.4f}, Validation Recon Loss: {val_recon_loss:.4f}, Validation Classification Loss: {val_classification_loss:.4f}')
            print(f'Accuracy: {accuracy_train:.4f}, Precision: {precision_train:.4f}, Recall: {recall_train:.4f}, F1: {f1_train:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, Validation F1: {val_f1:.4f}')
            print('---')
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Loss: {avg_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, Classification Loss: {avg_classification_loss:.4f}')
            print(f'Accuracy: {accuracy_train:.4f}, Precision: {precision_train:.4f}, Recall: {recall_train:.4f}, F1: {f1_train:.4f}')
            print('---')
            

    return history_train, history_val, history_metrics_train, history_metrics_val