import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import os
import sys
import time
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# --- IMPORTACIÓN DE LA NUEVA RED ---
from mamba_model import NeuroBackMamba
from data import MyOwnDataset, SortedBucketSampler

# Configuración de argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['pretrain', 'finetune'])
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--layers', type=int, default=12)
args = parser.parse_args()

# Configuración de Rutas
BASE_LOG = f"./log/mamba_{args.mode}"
BASE_MODEL = f"./models/mamba_{args.mode}"
os.makedirs(BASE_LOG, exist_ok=True)
os.makedirs(BASE_MODEL, exist_ok=True)

# Configuración de Dispositivo y Semilla
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(77)

# --- CARGA DE DATOS ---
dataset_path = f"./data/pt/{args.mode}" 
dataset_train = MyOwnDataset(root=dataset_path)
dataset_vld = MyOwnDataset(root='./data/pt/validation')

batch_size_vld = 2

# sampler_train = SortedBucketSampler(dataset_train, args.batch_size, shuffle=True)
# sampler_vld = SortedBucketSampler(dataset_train, batch_size_vld, shuffle=False)
#
# train_loader = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler_train, num_workers=12, pin_memory=False)
# vld_loader = DataLoader(dataset_vld, batch_size_vld, sampler=sampler_vld, num_workers=4, pin_memory=False)

train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True)
vld_loader = DataLoader(dataset_vld, batch_size_vld, shuffle=False, num_workers=4, pin_memory=True)

# --- INICIALIZACIÓN DEL MODELO ---
input_dim = dataset_train.num_node_features 

model = NeuroBackMamba(
    input_dim=input_dim,
    hidden_dim=args.hidden_dim,
    num_layers=args.layers,
    dropout=0.1
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
# Usamos el nuevo scaler recomendado para evitar warnings
scaler = torch.amp.GradScaler('cuda') 

if args.mode == 'finetune':
    ckpt_path = "./models/pretrain/pretrain-best.ptg"
    if os.path.exists(ckpt_path):
        print(f"Cargando pesos pre-entrenados desde {ckpt_path}")
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Advertencia: No se encontró checkpoint de pretrain.")

# --- FUNCIÓN DE ENTRENAMIENTO CORREGIDA ---
def train_epoch(epoch_idx, log_file):
    model.train()
    total_loss = 0
    total_samples = 0
    
    all_targets = []
    all_preds = []

    pbar = tqdm(train_loader, desc=f"Train Ep {epoch_idx}")
    
    for data in pbar:
        if data.y is None: continue
        
        data = data.to(device)
        
        # --- CORRECCIÓN CRÍTICA ---
        # 1. Encontramos los índices RELATIVOS dentro de data.y que son válidos (!= 2)
        # Esto nos da posiciones como [0, 1, 5, ...] hasta len(data.y)
        y_valid_indices = (data.y != 2).nonzero(as_tuple=True)[0]
        
        if y_valid_indices.numel() == 0: continue
        
        # 2. Obtenemos las etiquetas reales usando esos índices
        y_valid = data.y[y_valid_indices].float()
        
        # Cálculo de pesos para desbalance
        n_zeros = (y_valid == 0).sum()
        n_ones = (y_valid == 1).sum()
        pos_weight = (n_zeros + 1) / (n_ones + 1)
        
        # Reducción manual para aplicar pesos
        criterion = nn.BCEWithLogitsLoss(reduction='none') 
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            try:
                # El modelo devuelve predicciones para TODOS los nodos (ej. 1040)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                
                # --- CORRECCIÓN CRÍTICA ---
                # 3. Asumimos que data.y corresponde a los PRIMEROS nodos de data.x
                # (Esta es la convención estándar cuando len(y) < len(x) en grafos SAT)
                # Usamos los mismos índices válidos para extraer las predicciones correspondientes.
                # Como out es [1040, 1], out[y_valid_indices] extrae los valores correctos.
                pred_valid = out[y_valid_indices].view(-1)
                
                # Chequeo de seguridad dimensional
                if pred_valid.shape != y_valid.shape:
                    # Caso extremo: Si los índices exceden el tamaño de out (raro)
                    print(f"Error dimensional: Pred {pred_valid.shape} vs Target {y_valid.shape}")
                    continue

                loss_elements = criterion(pred_valid, y_valid)
                weights = torch.ones_like(y_valid)
                weights[y_valid == 1] = pos_weight
                loss = (loss_elements * weights).mean()
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * y_valid.size(0)
        total_samples += y_valid.size(0)
        
        with torch.no_grad():
            preds_cls = (pred_valid >= 0.5).long()
            all_targets.append(y_valid.cpu())
            all_preds.append(preds_cls.cpu())

    if len(all_targets) > 0:
        y_true = torch.cat(all_targets).numpy()
        y_pred = torch.cat(all_preds).numpy()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        log_msg = f"Epoch {epoch_idx} Loss: {avg_loss:.4f} F1: {f1:.4f}\n"
        log_file.write(log_msg)
        print(log_msg.strip())
        return f1
    return 0.0

# --- FUNCIÓN DE EVALUACIÓN CORREGIDA ---
def evaluate(log_file):
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for data in tqdm(vld_loader, desc="Validating"):
            try:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    data = data.cpu()
                    model_cpu = model.to('cpu')
                    out = model_cpu(data.x, data.edge_index, data.edge_attr, data.batch)
                    model.to(device)
                else:
                    raise e
            
            # --- LÓGICA DE ÍNDICES TAMBIÉN AQUÍ ---
            y_valid_indices = (data.y != 2).nonzero(as_tuple=True)[0]
            if y_valid_indices.numel() == 0: continue
            
            pred_valid = out[y_valid_indices].view(-1)
            y_valid = data.y[y_valid_indices].float()
            
            preds_cls = (pred_valid >= 0).long()
            all_targets.append(y_valid.cpu())
            all_preds.append(preds_cls.cpu())

    if len(all_targets) > 0:
        y_true = torch.cat(all_targets).numpy()
        y_pred = torch.cat(all_preds).numpy()
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        matrix = confusion_matrix(y_true, y_pred)
        
        log_file.write(f"Validation F1: {f1:.4f}\nConfusion Matrix:\n{matrix}\n")
        return f1
    return 0.0

# --- BUCLE PRINCIPAL ---
best_f1 = 0.0
log_path = os.path.join(BASE_LOG, f"training.log")

with open(log_path, "a") as log_file:
    for epoch in range(args.epochs):
        train_f1 = train_epoch(epoch, log_file)
        val_f1 = evaluate(log_file)
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'f1': val_f1
        }
        
        torch.save(save_dict, os.path.join(BASE_MODEL, "last.ptg"))
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(save_dict, os.path.join(BASE_MODEL, f"{args.mode}-best.ptg"))
            print(f"¡Nuevo récord! F1: {best_f1:.4f}")

print("Entrenamiento finalizado.")
