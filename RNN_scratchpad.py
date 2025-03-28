# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 16:59:54 2025

@author: dowel
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from analysis_funs.CX_behaviour_pred_col import CX_b
import matplotlib.pyplot as plt 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import os
plt.rcParams['pdf.fonttype'] = 42 
from Utilities.utils_general import utils_general as ug

#%% 
datadirs = [
    "Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f1\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240418\\f2\\Trial3",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240502\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241104\\f1\\Trial5",
"Y:\Data\FCI\Hedwig\FC2_maimon2\\241106\\f1\\Trial2",
r"Y:\Data\FCI\Hedwig\FC2_maimon2\250128\f1\Trial4",
r"Y:\Data\FCI\Hedwig\FC2_maimon2\250220\f1\Trial2"#Training
]
#%%
datadir = datadirs[5]
cxb = CX_b(datadir,['eb','fsb_upper','fsb_lower'])
cxb.prep4RNN(2,['eb'],'Phase_amp',downsample=True,downfactor=3)
#cxb.prep4RNN(2,['fsb_upper'],'wedges',downsample=True,downfactor=3)

X = cxb.input_mat.astype('float32')
yo = cxb.y.astype('float32')
yo = yo[:,np.newaxis]

y = np.append(np.sin(yo),np.cos(yo),axis=1)
y = y[cxb.winlen:,:]

#%%


torch.manual_seed(42)
samples = 1000
timesteps = 30
features = np.shape(X)[2]
output_dim = 2
num_layers = 2
units = 25
#X = np.random.randn(samples, timesteps, features).astype(np.float32)
#y = np.random.randn(samples, output_dim).astype(np.float32)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size,num_layers=num_layers, batch_first=True,dropout=0.3)
        self.linear1 = nn.Linear(hidden_size, 1)
        self.rnn2 = nn.LSTM(1, hidden_size,num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        
    
    def forward(self, x,return_low=False):
        
        out, _ = self.rnn(x)
        out = self.linear1(out)
        if return_low:
           return out
        out,_ = self.rnn2(out)
        out = self.fc(out[:, -1, :])  # Take last timestep output
        return out
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleRNN(features, units, output_dim,num_layers).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 50
batch_size =16

tscv = TimeSeriesSplit(n_splits=5)
losses = []
#%% Train on first 10 entries and test on rest

instrip = cxb.instrip[cxb.winlen:]
blocks,blocksize = ug.find_blocks(instrip)
first10 = blocks[:50]
train_idx = np.arange(0,first10[-1])
test_idx = np.arange(0,len(instrip))
test_idx = test_idx[~np.isin(test_idx,train_idx)]
#%%
plt.close('all')
X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
y_train, y_test = y_tensor[train_idx].view(-1,2), y_tensor[test_idx].view(-1,2)

# Initialize model, loss, and optimizer for each fold
model = SimpleRNN(features, 50, output_dim,num_layers).to(device)
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)#, weight_decay=0.01) # With weight decay becomes L2
epochs = 200
l1_lambda = 0.01

for epoch in range(epochs):
    permutation = torch.randperm(X_train.shape[0])
    epoch_loss = 0
    
    for i in range(0, X_train.shape[0], batch_size):
        indices = permutation[i:i + batch_size]
        #X_batch, y_batch = torch.tensor(X[indices], dtype=torch.float32), torch.tensor(y[indices], dtype=torch.float32)
        X_batch = X_train[indices].to(device)
        y_batch = y_train[indices,:].to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # L1 regularization term
        #l1_penalty = np.sum(p.abs().sum() for p in model.parameters())
        #loss = loss + l1_lambda * l1_penalty
        
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        epoch_loss += loss.item()

    avg_loss = epoch_loss / (X_tensor.shape[0] // batch_size)
    plt.figure(101)
    plt.scatter(epoch,avg_loss,color='k')
    plt.show()
    
    plt.figure(102)
    y_pred = model(X_test.to(device)).cpu()
    val_loss = criterion(y_pred, y_test).item()
    plt.scatter(epoch,val_loss,color='k')
    plt.show()
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Validation loss: {val_loss:.4f}')
        
    
# Evaluate on validation set
with torch.no_grad():
    y_pred = model(X_test.to(device)).cpu()
    val_loss = criterion(y_pred, y_test).item()
    #print(f'Validation Loss for Fold {fold + 1}: {val_loss:.4f}')
    losses.append(val_loss)



xpred = X_tensor.to(device)
y_pred = model(xpred).cpu().detach().numpy()
ylo = model.forward(xpred,return_low=True).cpu().detach().numpy()
plt.figure()
yp = np.arctan2(y_pred[:,0],y_pred[:,1])
#yp2 = np.arctan2(y_pred[:,1],y_pred[:,0])
ld = len(yo)-len(yp)
t = cxb.tt
plt.plot(t,yo[:,0],color='k')
plt.plot(t[cxb.winlen:],yp,color='g')
#plt.plot(t[:-ld],ylo[:,-1,0],color='b')
plt.plot(t[cxb.winlen:],cxb.instrip[cxb.winlen:],color='r')
plt.scatter(t[train_idx],t[train_idx]*0+0.1,color=[1,0.5,0.2],s =1,zorder=10)
#%% Train on successively larger portions
for fold, (train_idx,test_idx) in enumerate(tscv.split(X)):
    print(f'\nFold {fold + 1}/{tscv.get_n_splits()}')

    # Split data
    X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
    y_train, y_test = y_tensor[train_idx].view(-1,2), y_tensor[test_idx].view(-1,2)
    
    # Initialize model, loss, and optimizer for each fold
    model = SimpleRNN(features, 50, output_dim,num_layers).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.shape[0])
        epoch_loss = 0
        
        for i in range(0, X_train.shape[0], batch_size):
            indices = permutation[i:i + batch_size]
            #X_batch, y_batch = torch.tensor(X[indices], dtype=torch.float32), torch.tensor(y[indices], dtype=torch.float32)
            X_batch = X_train[indices].to(device)
            y_batch = y_train[indices,:].to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            epoch_loss += loss.item()
    
        avg_loss = epoch_loss / (X_tensor.shape[0] // batch_size)
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
    # Evaluate on validation set
    with torch.no_grad():
        y_pred = model(X_test.to(device)).cpu()
        val_loss = criterion(y_pred, y_test).item()
        print(f'Validation Loss for Fold {fold + 1}: {val_loss:.4f}')
        losses.append(val_loss)



    xpred = X_tensor.to(device)
    y_pred = model(xpred).cpu().detach().numpy()
    ylo = model.forward(xpred,return_low=True).cpu().detach().numpy()
    plt.figure()
    yp = np.arctan2(y_pred[:,0],y_pred[:,1])
    #yp2 = np.arctan2(y_pred[:,1],y_pred[:,0])
    ld = len(yo)-len(yp)
    t = np.arange(0,len(yo))
    plt.plot(t,yo[:,0],color='k')
    plt.plot(t[:-ld],yp,color='g')
    plt.plot(t[:-ld],ylo[:,-1,0],color='b')
    plt.plot(t,cxb.instrip,color='r')
    plt.scatter(t[train_idx],t[train_idx]*0+0.1,color=[1,0.5,0.2],s =1,zorder=10)
    plt.title('Fold: '+str(fold) + ' loss: ' +str(np.round(val_loss,decimals=2)))
    #plt.plot(cxb.cxa.ft2['instrip'],color='r')
    plt.savefig(os.path.join('Y:\Data\FCI\FCI_summaries\PhaseML\FC2_test','fold'+str(fold)+'.pdf'))
    
    
#%%
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

for fold, (train_idx,test_idx) in enumerate(tscv.split(X)):
    print(f'\nFold {fold + 1}/{tscv.get_n_splits()}')

    # Split data
    X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
    y_train, y_test = y_tensor[train_idx].view(-1,1), y_tensor[test_idx].view(-1,1)

    # Initialize model, loss, and optimizer for each fold
    model = SimpleRNN(features, 50, output_dim,num_layers).to(device)
    criterion = nn.MSELoss()
    
    
    # Training loop
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.shape[0])
        epoch_loss = 0

        for i in range(0, X_train.shape[0], batch_size):
            indices = permutation[i:i + batch_size]
            X_batch, y_batch = X_train[indices], y_train[indices]
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(loss)
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (X_train.size(0) // batch_size)
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

    # Evaluate on validation set
    with torch.no_grad():
        y_pred = model(X_test.to(device)).cpu()
        val_loss = criterion(y_pred, y_test).item()
        print(f'Validation Loss for Fold {fold + 1}: {val_loss:.4f}')
        losses.append(val_loss)

# Final average loss across folds
mean_loss = np.mean(losses)
print(f'\nAverage Cross-Validation Loss: {mean_loss:.4f}')
    
    
    
    
    
#%%    

#%% Working version - over trained



for epoch in range(epochs):
    permutation = torch.randperm(X_tensor.shape[0])
    epoch_loss = 0
    
    for i in range(0, X_tensor.shape[0], batch_size):
        indices = permutation[i:i + batch_size]
        #X_batch, y_batch = torch.tensor(X[indices], dtype=torch.float32), torch.tensor(y[indices], dtype=torch.float32)
        X_batch = X_tensor[indices].to(device)
        y_batch = y_tensor[indices].to(device)
        y_batch = y_batch.view(-1,1)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / (X_tensor.shape[0] // batch_size)
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')



xpred = X_tensor.to(device)
y_pred = model(xpred).cpu().detach().numpy()


plt.plot(y_pred)
plt.plot(y)

#%%
















for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Predict on new data
#X_new = np.random.randn(1, timesteps, features).astype(np.float32)
X_new_tensor = torch.tensor(X)
y_pred = model(X_new_tensor).detach().numpy()
print("Predicted y:", y_pred)




# for epoch in range(epochs):
#     permutation = torch.randperm(X.shape[0])
#     for i in range(0, X.shape[0], batch_size):
#         indices = permutation[i:i + batch_size]
#         X_batch, y_batch = X[indices], y[indices]
        
#         # Forward pass
#         outputs = model(X_batch)
#         loss = criterion(outputs, y_batch)
        
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    
#     if (epoch + 1) % 5 == 0:
#         print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# # Prediction
# with torch.no_grad():
#     y_pred = model(X)

# # Print a sample of predictions
# print("Predictions:", y_pred[:5])