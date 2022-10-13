import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import math
from matplotlib import pyplot
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(">> My model uses {}.".format(device))
torch.manual_seed(0)
np.random.seed(0)

# BEST : 21 / (2, 3)
calculate_loss_all_values = False
input_window = 21     # num of input steps (input days)
output_window = 2     # num of prediction multi steps (intuite & calculate loss output days)  
batch_size = 32


#####################################
## 1. Positional Encoding function ##
#####################################
class PositionalEncoding(nn.Module):
    """
     src = positional encoding + embedding vector
     PE(pos, 2i) = sin(pos / 10000**(2i / d_model)
     PE(pos, 2i+1) = cos(pos / 10000**(2i / d_model)
    """
    def __init__(self, d_model, max_len = 5000):
        super(PositionalEncoding, self).__init__()

        PE = torch.zeros(max_len, d_model)                                          # (5000, 512)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)       # (5000, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        PE[:, 0::2] = torch.sin(position * div_term)   # index = 2_i+1
        PE[:, 1::2] = torch.cos(position * div_term)   # index = 2_i
        PE = PE.unsqueeze(0).transpose(0, 1)           # (5000, 1, 512)
        self.register_buffer('PE', PE)
    
    def forward(self, x):
        return x + self.PE[:x.size(0), :]   # Embedding vector + Positional Encoding

###########################
## 2. Transformer models ##
###########################
class MyTransformer(nn.Module):
    """
    Pos-Encoding : PositionalEncoding Class
    Encoder : Transformer Encoder + Look ahead masking
    Decoder : Linear model
    """
    def __init__(self, feature_size = 512, num_layers = 3, dropout = 0.05):
        super(MyTransformer, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        # Encoder : Transformer Encoder model including MultiheadAttention(nhead = the number of heads in the multi-head Attention models)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = feature_size, nhead = 8, dropout = dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_layers)
        # Decoder : Linear model((N, 512) -> (N, 1))
        self.decoder = nn.Linear(feature_size, 1)
        # Initialize
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)
    
    # Look Ahead masking
    def _generate_look_ahead_mask(self, seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        # 1. Encoder masking
        if self.src_mask == None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_look_ahead_mask(len(src)).to(device)
            self.src_mask = mask 
        # 2. Pos-Encoding
        src = self.pos_encoder(src)
        # 3. Encoder
        output = self.transformer_encoder(src, self.src_mask)
        # 4. Decoder
        output = self.decoder(output)
        return output

##############################
## 3. DataLoader functions ###
##############################
def create_sequences(input_data, window_size):
    seqs = []
    input_len = len(input_data)
    for i in range(input_len - window_size):
        # Multi train & prediction
        train_seq = np.append(input_data[i : i + window_size][: -output_window] , output_window * [0])
        train_label = input_data[i : i + window_size]
        seqs.append((train_seq, train_label))
    return torch.FloatTensor(seqs)

def get_data(df, tgt_col):
    # 1. Get data
    my_data = df[tgt_col].values
    
    # 2. Normalization
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    my_data = scaler.fit_transform(my_data.reshape(-1, 1)).reshape(-1)
    
    # 3. Hold-out split(Train : Test = 90 : 10)
    samples = round(len(my_data) * 0.90)
    train_data = my_data[:samples]
    test_data = my_data[samples:]

    # 4. Create sequence data
    train_sequence = create_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]            

    test_sequence = create_sequences(test_data, input_window)
    test_sequence = test_sequence[:-output_window]

    return train_sequence.to(device), test_sequence.to(device)  # Train & Test : (seq_len, (seq, label), window_size)

def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))  # feature size : 1 (Temperture only one)
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target

##############################################
## 4. Model training & evaluation functions ##
##############################################
def train(train_data, model, criterion, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        # Multi train & prediction
        if calculate_loss_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(f"{epoch:^8} | {batch:^8} | {scheduler.get_last_lr()[0]:^10f} | {elapsed * 1000 / log_interval:^10f} | {cur_loss:^10f} | {math.exp(cur_loss):^10f}")

            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source, criterion):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)
            # Multi train & prediction
            if calculate_loss_all_values:
                total_loss += len(data[0])* criterion(output, targets).cpu().item()
            else:                                
                total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()

        return total_loss / len(data_source)


def plot_and_loss(eval_model, criterion, data_source, epoch):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i,1)
            output = eval_model(data)
            # Multi train & prediction
            if calculate_loss_all_values:                                
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()

            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
             
    len(test_result)
    pyplot.figure(figsize = (10, 6))
    pyplot.plot(test_result,color = "C3", label = 'Prediction')               # Prediction val
    pyplot.plot(truth[:500],color = "C0", label = 'True')                     # True val
    pyplot.plot(test_result-truth,color = "silver", label = 'Residual')       # Residual
    pyplot.grid(True, which = 'both')
    pyplot.axhline(y = 0, color = 'k')
    pyplot.legend(loc = 'upper right')
    pyplot.show()
    pyplot.close()
    
    return total_loss / i

# predict the next n steps based on the input data 
def predict_future(eval_model, data_source, steps):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    data, _ = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps):            
            output = eval_model(data[-input_window:])                        
            data = torch.cat((data, output[-1:]))
            
    data = data.cpu().view(-1)

    pyplot.figure(figsize = (10, 6))
    pyplot.plot(data, color = "C3", linestyle = '--', label = 'Forcast')  # Neg : C3, Pos : C9
    pyplot.plot(data[:input_window],color = "C0", label = 'Input data')          
    pyplot.grid(True, which = 'both')
    pyplot.axhline(y=0, color = 'k')
    pyplot.legend(loc = 'upper right')
    pyplot.show()
    pyplot.close()

def train_eval(train_data, val_data, model, criterion, optimizer, scheduler, epochs, prediction_steps):
    print("\n ******* My Transformer model is Training ... ******* \n")
    best_val_loss = float("inf")
    best_model = None
    print(f"{'Epoch':^8} | {'Batches':^8} | {'LR':^10} | {'Time (ms)':^10} | {'Train Loss':^10} | {'PPL':^10}")
    print("="*72)
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_data, model, criterion, optimizer, scheduler, epoch)

        if(epoch % 100 == 0):
            print('-' * 72)
            print("\n")
            print("******* Prediction Graph *******")
            print("\n")
            # show validation loss
            val_loss = plot_and_loss(model, criterion, val_data, epoch)
            print("\n")
            # predict N STEPS trained models
            predict_future(model, val_data, prediction_steps)
            print("\n")
        else:
            val_loss = evaluate(model, val_data, criterion)

        print('-' * 72)
        print('>> Validation({:5.2f}s)   :   Val Loss : {:5.5f}     Val PPL : {:5.5f}'.format((time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        print('-' * 72)       
    print('=' * 72)
    print("******* Training is Done ! *******")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
    
    scheduler.step()