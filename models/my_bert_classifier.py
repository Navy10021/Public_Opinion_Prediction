import torch
import torch.nn as nn
from transformers import AutoModel

######################################
## BERT-based Text classifier model ##
######################################

class MyBERT_Classifier(nn.Module):
    def __init__(self, model_name, freeze_bert = False, embedding_type = "cls"):
        """
        model_name : Pre-trained language models
        embedding_type : 'cls', 'max', 'mean'
        """
        super(MyBERT_Classifier, self).__init__()

        D_in, H, D_out = 768, 64, 2      # [Hidden size of BERT, Hidden size of our classifier, Number of labels]
        
        self.emb_type = embedding_type
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.25),    
            nn.Linear(H, D_out),
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, att_masks):
        # 1.Token-based Fixed vector representation
        outputs = self.bert(input_ids = input_ids, attention_mask = att_masks)
        embeddings = outputs[0]                               # Token Embeddings : [num_text, token_max_length(512), PLM_hidden_dim(768)]
        mask = att_masks.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        
        # 2. Extract Text Embeddings (Three methods)
        # 2-1. Using [CLS] token 
        if self.emb_type == 'cls':
            # last hidden state [CLS]
            text_embeddings = masked_embeddings[:, 0, :]
            text_embeddings = text_embeddings.squeeze(0)      # [CLS] Toekn : [num_text, PLM_hidden_dim(768)] 
        # 2-2. Using Max Pooling
        elif self.emb_type == 'max':
            masked_embeddings[masked_embeddings == 0] = -1e9
            text_embeddings = torch.max(masked_embeddings, 1)[0]
            text_embeddings = text_embeddings.squeeze(0)      # Max Pooling Embeddings : [num_text, PLM_hidden_dim(768)] 
        # 2-3. Using Mean Pooling
        else:
            sum_emb = torch.sum(masked_embeddings, 1)
            counted = torch.clamp(mask.sum(1), min = 1e-9)
            text_embeddings = sum_emb / counted
            text_embeddings = text_embeddings.squeeze(0)      # Mean Pooling Embeddings  : [num_text, PLM_hidden_dim(768)] 
            
        # 3. Compute logits
        logits = self.classifier(text_embeddings)

        return logits
    

##########################
## model Initialization ##
##########################
from transformers import AdamW, get_linear_schedule_with_warmup

device = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_model(model_name, epochs):
    
    bert_classifier = MyBERT_Classifier(
        model_name,
        freeze_bert = False,
        embedding_type = "mean"         # cls, mean, max
        )
    bert_classifier.to(device)

    optimizer = AdamW(
        bert_classifier.parameters(),
        lr = 1e-5,
        eps = 1e-7
        )
    
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps
        )
    
    return bert_classifier, optimizer, scheduler