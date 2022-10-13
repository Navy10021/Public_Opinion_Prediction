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
  STEP 1) Sentiment analysis corpus preparation: positive/negative labeled Twitter and comment text.
  STEP 2) Fine-tuning a text embedding classification model: learn a BERT-based language model designed to create and classify text embeddings (or vectors) in three ways.
  STEP 3) Time series data conversion: Convert the positive/negative predictive values of the language model into a time series table.
  STEP 4) Applying a Transformers-based time series prediction model: Predict the future trend of positive/negative public opinion after learning with my time series data prediction model designed based on Transformers.
  
![task_1_overall](https://user-images.githubusercontent.com/105137667/195532709-3071aee0-e6db-481a-b97f-220e39e540fa.jpg)

