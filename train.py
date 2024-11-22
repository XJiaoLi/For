import os
from time import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

print(torch.__version__)
print(torch.cuda.is_available())
import torch.nn as nn
import torch.utils.data as Data
from utils import (generate_data, masked_mae_np, masked_mape_np, masked_mse_np, compute_val_loss, get_adjacency_matrix)
from ourmodel import make_model
import torch.optim as optim

'''
#画图方式1
def plot_attention_heatmap(matrix, title):
    matrix_slice = matrix[0,:,:,1]
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_slice, annot=False, fmt=".2f", cmap='viridis')
    plt.title(title)
    plt.xlabel('Head')
    plt.ylabel('Position')
    plt.show()

#画图方式2
def plot_attention_heatmap2(matrix, title):
    # 对时间步和批次进行平均，得到 (170, 32) 的二维矩阵
    matrix_slice1 = np.mean(matrix, axis=(0, 1))
    # 对节点和批次进行平均，得到 (12, 32) 的二维矩阵
    matrix_slice2 = np.mean(matrix, axis=(0, 2))

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_slice1, annot=False, fmt=".2f", cmap='viridis')
    plt.title(title)
    plt.xlabel('Feature')
    plt.ylabel('Node' )
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_slice2, annot=False, fmt=".2f", cmap='viridis')
    plt.title(title)
    plt.xlabel('Feature')
    plt.ylabel('Time Step')
    plt.show()
'''

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file',default='config/PEMS08/PEMS08_32.json')
parser.add_argument("--save", action="store_true", help="save model")
args = parser.parse_args()

config_filename = args.config

with open(config_filename, 'r') as f:
    config = json.loads(f.read())
    
print(json.dumps(config, sort_keys=True, indent=4))

batch_size = config['batch_size']
learning_rate=config['learning_rate']
num_of_vertices = config['num_of_vertices']
graph_signal_matrix_filename = config['graph_signal_matrix_filename']
ctx=config['ctx']
nb_block=config['nb_block']
nb_filter=config['num_of_filter']
K=config['K']
len_input=config['len_input']
in_channels=config['num_of_features']
num_for_predict=config['num_for_predict']
os.environ['CUDA_VISIBLE_DEVICES']='0'
DEVICE = torch.device('cuda:0')
spa_adj_filename=config['spa_adj_filename']
sem_adj_filename=config['sem_adj_filename']
model_path=config['model_path']
epochs=config['epochs']
id_filename=config['id_filename']
loaders = []
true_values = []

for idx, (x, y) in enumerate(generate_data(graph_signal_matrix_filename,DEVICE)):
  loaders.append(Data.DataLoader(
  dataset=Data.TensorDataset(x,y),
  batch_size=batch_size,
  shuffle=(idx == 0),
  num_workers=0
)
  )

  if idx == 0:
    training_samples = x.shape[0]
    train_values=x
  else:
    true_values.append(y)

train_loader, val_loader, test_loader = loaders
val_y, test_y = true_values

spatial_adj, distace_adj =get_adjacency_matrix(spa_adj_filename,num_of_vertices,id_filename)

sem_adj = torch.load(sem_adj_filename)

net = make_model(DEVICE, nb_block, in_channels, nb_filter, num_for_predict, len_input, num_of_vertices,K,spatial_adj,sem_adj)

def training():
  if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']
    print('load epoch {} successful!'.format(start_epoch))
  else:
    start_epoch = 0
  
  total_param = 0
  criterion = nn.SmoothL1Loss().to(DEVICE)
  optimizer = optim.Adam(net.parameters(), lr=learning_rate)
  global_step = 0
  best_epoch = 0
  best_val_loss = np.inf
  start_time = time()
  for epoch in range(start_epoch, start_epoch+epochs):
    val_loss = compute_val_loss(net, val_loader, criterion, epoch)
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_epoch = epoch
      with torch.no_grad():
        prediction = []
        loader_length = len(test_loader)
        for step, (b_x,b_y) in enumerate(test_loader):
          outputs = net(b_x)
          prediction.append(outputs.detach().cpu().numpy())
          if step % 10 == 0:
            print('epoch: %s  predicting data set batch %s / %s' % (epoch, step + 1, loader_length))
      prediction = np.concatenate(prediction, 0)
        
      mae = masked_mae_np(test_y.reshape(-1, 1).detach().cpu().numpy(),prediction.reshape(-1, 1),0)
      rmse = masked_mse_np(test_y.reshape(-1, 1).detach().cpu().numpy(),prediction.reshape(-1, 1),0)** 0.5
      mape = masked_mape_np(test_y.reshape(-1, 1).detach().cpu().numpy(),prediction.reshape(-1, 1),0)
      print('MAE: %.2f' % (mae))
      print('RMSE: %.2f' % (rmse))
      print('MAPE: %.2f' % (mape))  
      
      state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(),'epoch': epoch}
      torch.save(state, model_path)
    
    net.train()
    for step, (b_x,b_y) in enumerate(train_loader):
      optimizer.zero_grad()
      outputs = net(b_x)
      loss = criterion(outputs, b_y)
      loss.backward()
      optimizer.step()
      training_loss = loss.item()
      global_step += 1
      if global_step % 1000 == 0:
        print('global step: %s, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time() - start_time))

  print('**** testing model ****')
  print('best epoch:', best_epoch)
  net.train(False)
  with torch.no_grad():
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model'])
    prediction = []
    loader_length = len(test_loader)
    for step, (b_x,b_y) in enumerate(test_loader):
      outputs = net(b_x)
      prediction.append(outputs.detach().cpu().numpy())
      if step % 10 == 0:
         print('predicting data set batch %s / %s' % (step + 1, loader_length))
         
    prediction = np.concatenate(prediction, 0)
    mae = masked_mae_np(test_y.reshape(-1, 1).detach().cpu().numpy(),prediction.reshape(-1, 1),0)
    rmse = masked_mse_np(test_y.reshape(-1, 1).detach().cpu().numpy(),prediction.reshape(-1, 1),0)** 0.5
    mape = masked_mape_np(test_y.reshape(-1, 1).detach().cpu().numpy(),prediction.reshape(-1, 1),0)
    print('MAE: %.2f' % (mae))
    print('RMSE: %.2f' % (rmse))
    print('MAPE: %.2f' % (mape))
     
if __name__ == '__main__':
  training()
