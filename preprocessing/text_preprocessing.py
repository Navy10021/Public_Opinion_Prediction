import re
import emoji
import hanja
import torch

# 1. Filtering Chinese characters
def cleansing_chinese(sent):
    # Chinese to Korean
    sent = hanja.translate(sent, 'substitution')
    return sent

# 2. Clean emojis
def clean_emoji(text):
    return emoji.replace_emoji(text)

# 3. Clean multi space
def clean_multi_space(text):
    text = str(text)
    return re.sub("\s\s+" , " ", text)

# 4. Filtering special characters and spaces
def cleansing_special(sent):
    sent = re.sub("[](),,ㆍ·\'\"’‘”“!?\\‘|\<\>`\'[\◇….]", " ", sent)
    sent = re.sub("[^가-힣a-zA-Z0-9\\s]", " ", sent)
    sent = re.sub("\s+", " ", sent)
    sent = sent.strip()
    return sent

# 5. Final Preprocessing
def cleansing_sent(sent):
    clean_sent = cleansing_chinese(sent)
    clean_sent = cleansing_special(clean_sent)
    clean_sent = clean_emoji(clean_sent)
    clean_sent = clean_multi_space(clean_sent)
    return clean_sent

# 6. Kor Text preprocessing samples
sentence = '[1] 文대통령이 "실언"했다는      北김여정…아슬아슬한 (남북관계).'
clean_sentence = cleansing_sent(sentence)

print(">> Before Preprocessing : {}".format(sentence))
print(">> After Preprocessing : {}".format(clean_sentence))



# 7. preprocessing for BERT 
#plm_model = 'klue/roberta-base'
#tokenizer = AutoTokenizer.from_pretrained(plm_model)

def preprocessing_for_bert(data, max_len = 128):
    # 1. List to store outputs
    input_ids, att_masks = [], []
    for i in range(len(data)):
        encoded_sent = tokenizer.encode_plus(
            data[i],                           # Cleaned text
            add_special_tokens = True,         # Add [CLS], [SEP]
            max_length = max_len,              # Max length
            pad_to_max_length = True,          # The same size for MAX_LEN : Cut at this length or zero-pad
            return_attention_mask = True,      # Return attention mask
        )
    
        # Add the outputs to List
        input_ids.append(encoded_sent.get('input_ids'))
        att_masks.append(encoded_sent.get('attention_mask'))
    
    # Convert list -> Tensor
    input_ids = torch.tensor(input_ids)
    att_masks = torch.tensor(att_masks)

    return input_ids, att_masks