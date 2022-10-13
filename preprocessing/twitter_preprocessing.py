#! pip install emoji
#! pip install hanja

import numpy as np
import pandas as pd
import os
import re, string
import emoji
import hanja


# 1. Filtering Chinese characters
def clean_chinese(sent):
    # Chinese to Korean
    sent = hanja.translate(sent, 'substitution')
    return sent


# 2. Clean emojis
def clean_emoji(text):
    return emoji.replace_emoji(text)


# 3. Clean text(Remove punctuations, links, mentions, \r\n)
def clean_text(text):
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ')
    text = re.sub(r"(?:\@|https?\://,)\S+", " ", text)
    text = re.sub('[^가-힣3\\s]', '', text)
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text


# 4. Clean hashtags at the end of the sentence
def clean_hashtage(text):
    text = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', text)) # Remove last hash
    text = " ".join(word.strip() for word in re.split('#|_', text))
    return text


# 5. Filter special characters($ and &)
def filter_chars(sent):
    new_sent = []
    for word in sent.split(' '):
        if ('$' in word) | ('&' in word):
            new_sent.append('')
        else:
            new_sent.append(word)
    return ' '.join(new_sent)


# 6. Clean multi space
def clean_multi_space(text):
    text = str(text)
    return re.sub("\s\s+" , " ", text)