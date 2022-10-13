import pandas as pd
from models.my_bert_classifier import *
from models.train_eval import *
from preprocessing.text_preprocessing import *
from torch.utils.data import TensorDataset, DataLoader


##################################################
# Pos / Neg prediction with Pre-trained My model #
##################################################

# 1. Load preprocessed text dataset(News + Twitter) 
df = pd.read_csv('./data/clean_twitter_1.csv', encoding='utf-8')
df = df.dropna()
df = df.sort_values(by="Date")

# 2. Extract prediction data(Twitter)
X_pred = df['Text'].values
print(">> Total Data size : ", len(X_pred))

# 3. Tokenizer
pred_input_ids, pred_att_masks = preprocessing_for_bert(X_pred)

# 4. DataLoader
batch_size = 32
pred_data = TensorDataset(pred_input_ids, pred_att_masks)
pred_dataloader = DataLoader(pred_data, batch_size = batch_size)

# 5. Positive/negative prediction through fine-tuned language model
plm_model = 'klue/roberta-base'
save_path = './outputs/sentiment_model_cls.pt'

model = MyBERT_Classifier(plm_model, freeze_bert = False, embedding_type = "cls")
model.load_state_dict(torch.load(save_path))
twitter_preds = model_predict(model, pred_dataloader)
print("\n Prediction complete ! \n")
print("\n\n >> Total Number of predictions : {}".format(len(twitter_preds)))

# 6. Create final prediction table
df['Predictions'] = twitter_preds
new_df = df[['Date','Text', 'Predictions']].set_index(keys = ['Date'], inplace = False, drop = True)


##########################################################
# Convert Sentiment analysis results to Time Series data #
##########################################################
def hash_table(dataframe):
    """
    Function which is making new dataframe is used to load data
    """
    result = {} # dict = {date : [(Negative, count), (Positive, count)]}
    tgt_keys = [0, 1]   # Positive, Negative
    # 1. Hashing function
    for i in sorted(list(set(dataframe.index))):
        if type(i) != str:
            date = i.strftime('%m-%d-%y')
        else:
            date = i
        val_count = dataframe[dataframe.index == i]['Predictions'].value_counts().to_dict()
        for j in tgt_keys:
            if j not in val_count:
                val_count[j] = 0
        val_count = sorted(val_count.items())
        result[date] = val_count

    Pos, Neg = [],[]
    for day in result:
        Neg.append(result[day][0][1])
        Pos.append(result[day][1][1])
    
    # 2. Make DataFrame
    count_df = pd.DataFrame({'Date' : list(result.keys()),
                             'Pos' : Pos,
                             'Neg' : Neg,
                             })
    count_df['Date'] = pd.to_datetime(count_df['Date'])
    count_df = count_df.set_index(keys = ['Date'], inplace = False, drop = True)

    # 3. Calculate Ration
    count_df['Total'] = count_df['Pos'] + count_df['Neg']
    count_df['Pos_ratio'] = count_df['Pos'] / count_df['Total']
    count_df['Neg_ratio'] = count_df['Neg'] / count_df['Total']

    return count_df

# Make DataFrame
data = hash_table(new_df)
print(">> Time series length :", len(data))
print(data.head())


##########
## Plot ##
##########
from matplotlib import pyplot
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "plotly_white"
plot_template = dict(
    layout=go.Layout({
        "font_size": 15,
        "xaxis_title_font_size": 15,
        "yaxis_title_font_size": 15}))

series = data[['Pos', 'Neg', 'Total']]
#series = data[['Pos_ratio', 'Neg_ratio']]
fig = px.line(series, labels = dict(created_at = "Date", value = "News & Public Opinion Trend", variable = ""))
fig.update_layout(template = plot_template,
                  legend = dict(orientation = 'h', y = 1.1, title_text = ""))
fig.show()


####################################################
## Transformer-based Time Series prediction model ##
####################################################
from models.Transformer_based_Predictor import *

# Target : Pos or Neg
target = "Neg" 
print(">> My model predicts '{} public opinion'".format(target))


train_data, val_data = get_data(series, target)
model = MyTransformer().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5, eps = 1e-7, weight_decay = 1e-3) # L2-Norm
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1.0, gamma = 0.1)  

#######################
## Train & Inference ##
#######################
train_eval(
    train_data = train_data,
    val_data = val_data,
    model = model,
    criterion = criterion,
    optimizer = optimizer,
    scheduler = scheduler,
    epochs = 1500,
    prediction_steps = 14,   # Predict days : 2 Weeks
    )