from preprocessing.text_preprocessing import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

#############
## Dataset ##
#############

# 1. DataLoad
train_data_path = 'data/train_data.csv'
test_data_path = 'data/test_data.csv'

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)
print(" >> Train data size :", len(train_df))
print(" >> Test data size :", len(test_df))
#train_df.tail()


# 2. Text Preprocessing
# Text dataset
df = train_df[['Text', 'Sentiment']]
test_df = test_df[['Text', 'Sentiment']]

# Clean train data
cleaned_text = []
for text in df.Text:
    new_text = cleansing_sent(text)
    cleaned_text.append(new_text)

# Clean test data
cleaned_test = []
for text in test_df.Text:
    new_text = cleansing_sent(text)
    cleaned_test.append(new_text)

df['cleaned_text'] = cleaned_text
test_df['cleaned_text'] = cleaned_test

# Dataset Text length
text_len = []
for text in df.cleaned_text:
    t_len = len(text.split())
    text_len.append(t_len)

df['text_len'] = text_len

test_text_len = []
for text in test_df.cleaned_text:
    t_len = len(text.split())
    test_text_len.append(t_len)

test_df['text_len'] = test_text_len

plt.figure(figsize=(20, 5))
ax = sns.countplot(x = 'text_len',
                   data = df[df['text_len'] < 60],
                   palette = 'RdYlGn')
plt.title('Training dataset Word Distribution')
plt.ylabel('Count')
plt.xlabel('')
plt.show()


# 3. Drop empty tweets and less tahn 5 words
min_words = 1
max_words = 36

df = df[df['text_len'] > min_words]
df = df[df['text_len'] < max_words]
test_df = test_df[test_df['text_len'] > min_words]
test_df = test_df[test_df['text_len'] < max_words]
print(df['Sentiment'].value_counts())

# Over Sampling for sample ratio
ros = RandomOverSampler()
# Train data
train_x, train_y = ros.fit_resample(np.array(df['cleaned_text']).reshape(-1, 1), np.array(df['Sentiment']).reshape(-1, 1))
train_os = pd.DataFrame(list(zip([x[0] for x in train_x], train_y)), columns =['cleaned_text', 'Sentiment'])

# Test data
test_x, test_y = ros.fit_resample(np.array(test_df['cleaned_text']).reshape(-1, 1), np.array(test_df['Sentiment']).reshape(-1, 1))
test_os = pd.DataFrame(list(zip([x[0] for x in test_x], test_y)), columns =['cleaned_text', 'Sentiment'])

print("\n ==== After Train dataset over sampling ==== \n")
print(train_os['Sentiment'].value_counts())

print("\n ==== After Test dataset over sampling ==== \n")
print(test_os['Sentiment'].value_counts())


# 4. Train : Validation : Test dataset
X = train_os['cleaned_text'].values
y = train_os['Sentiment'].values

# Train & validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                      test_size = 0.01,
                                                      stratify = y,
                                                      random_state = 42)
# Test
X_test = test_df['cleaned_text'].values
y_test = test_df['Sentiment'].values

print(">> Training data : {}".format(X_train.shape[0]))
print(">> Validation data : {}".format(X_valid.shape[0]))
print(">> Test data : {}".format(X_test.shape[0]))


###############
## Tokenizer ##
###############
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup

plm_model = 'klue/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(plm_model)

# Make token dataset(max_len = 128)
train_input_ids, train_att_masks = preprocessing_for_bert(X_train)
val_input_ids, val_att_masks = preprocessing_for_bert(X_valid)
test_input_ids, test_att_masks = preprocessing_for_bert(X_test)

################
## DataLoader ##
################
# 1. Convert label data to torch.Tensor
train_label = torch.tensor(y_train)
val_label = torch.tensor(y_valid)
test_label = torch.tensor(y_test)

# 2.DataLoader
batch_size = 32

train_data = TensorDataset(train_input_ids, train_att_masks, train_label)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)

val_data = TensorDataset(val_input_ids, val_att_masks, val_label)
val_sampler = RandomSampler(val_data)
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = batch_size)

test_data = TensorDataset(test_input_ids, test_att_masks, test_label)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

#################################################
## Train with BERT-based Text Classifier Model ##
#################################################
from models.my_bert_classifier import *
from models.train_eval import *

set_seed(42)
additional_train = False
train_epochs = 5
save_path = './outputs/sentiment_model_mean.pt'    # _mean.pt, _max.pt, _cls.pt

if additional_train: 
    model, optimizer, scheduler = initialize_model(model_name = plm_model, epochs = train_epochs)
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    train(model, train_dataloader, val_dataloader, epochs = train_epochs, save_path = save_path, evaluation = True)
else:
    model, optimizer, scheduler = initialize_model(model_name = plm_model, epochs = train_epochs)
    train(model, train_dataloader, val_dataloader, epochs = train_epochs, save_path = save_path, evaluation = True)

################
## Evaluation ##
################
from sklearn.metrics import classification_report
plm_model = 'klue/roberta-base'
save_path = './outputs/sentiment_model_mean.pt'

# CLS Token/Mean/Max Embeddings Prediction results
model = MyBERT_Classifier(plm_model, freeze_bert = False, embedding_type = "mean")
model.load_state_dict(torch.load(save_path))
test_preds = model_predict(model, test_dataloader)

# 10 Epochs(25% Dropout)
print('\n\n ===== Text Classification Report for Mean Pooling based - Embeddings ===== \n\n', 
      classification_report(y_test, test_preds, target_names = ['Negative', 'Positive']))