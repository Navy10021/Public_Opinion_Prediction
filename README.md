# Online Public Opinion Prediction Model

## 1. Project Background
  - The National Assembly of the Republic of Korea is making various efforts to detect and respond to public opinion on major social issues through opinion polls and the media.
  - However, there is a limit to the objective prediction, and considerable time and cost are incurred from preparing a solution to legislative connection. For example, in the 20th National Assembly, the approval rate of all bills was 13.2%, and the average processing time was 577.2 days.
  - Therefore, in this project, we propose a natural language processing-based artificial intelligence model that can efficiently predict online public opinion and is expected to be used as a policy decision tool in various fields.

## 2. Dataset
  - The Korean National Assembly provided articles, Twitter, and online community data related to major legislation in Korea.
  - Online comment and review data were additionally collected for fine-tuning the language model.

## 3. Overall pipeline
To summarize the entire process of the 『Online Public Opinion Prediction Model』 we designed, it consists of the following four steps.

  - STEP 1) Sentiment analysis corpus preparation: positive/negative labeled Twitter and comment text.
 
  - STEP 2) Fine-tuning a text embedding classification model: learn a BERT-based language model designed to create and classify text embeddings (or vectors) in three ways.
  
  - STEP 3) Time series data conversion: Convert the positive/negative predictive values of the language model into a time series table.
  
  - STEP 4) Applying a Transformers-based time series prediction model: Predict the future trend of positive/negative public opinion after learning with my time series data prediction model designed based on Transformers.
  
![task_1_overall](https://user-images.githubusercontent.com/105137667/195532709-3071aee0-e6db-481a-b97f-220e39e540fa.jpg)


### STEP 1. Sentiment analysis corpus preparation

  - A total of 530k text data, including legislative news and Twitter, provided by the Korea National Assembly, and online comments collected for sentiment analysis, were synthesized and pre-processed.
  
  - This text data was labeled (negative: 0, positive: 1) according to positive and negative public opinion.
  
  ![그림_1](https://user-images.githubusercontent.com/105137667/195534310-fdd01336-c5b1-4445-95b8-bdde82ab7339.jpg)
  ![그림_2](https://user-images.githubusercontent.com/105137667/195534324-999cbc31-5225-4c1e-a8d8-3a1a47b05ba8.jpg)

### STEP 2. Text Embedding Classifier
  
  - Using pre-trained BERT-based language models(PLMs), we can obtain a fixed-size contextual vector, which means Token Embedding.
  
  - Use the [CLS] token or apply a pooling technique to obtain sentence-level embedding instead of token-level embedding.
      
      1) [CLS] Token : A word-level vector containing the meaning of the entire token within a sentence.
      2) Mean Pooling : A sentence-level vector summarizing the semantic expression of all tokens.
      3) Max Pooling : A sentence-level vector summarizing the semantic expression of important tokens.
 
  ![sent_embedding](https://user-images.githubusercontent.com/105137667/195534528-a9e7373e-0570-44f3-a409-2f6bfa324a98.jpg)
