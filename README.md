# Sentiment Natural Language Processing Model
## Project Purpose
This project is to build a basic NLP model that can perform the sentiment analysis. Sentiment analysis, also known as opinion mining or emotion AI, uses the NLP model to identify and quantify the human emotions portraited from the given texts. This model is expected to provide quantified emotion level scales from 1 through 5, 1 being the most dissatisfied state, and 5 being the most satisfied state.
## Data Source
Data source was obtained from Kaggle, titled by [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
## Descriptive Statistics
Data is composed of 568,454 rows and 10 columns. Column "Text" contains the review texts, and "Score" contains the quantified scale of satisfaction from food, ranging from 1 through 5.
## Model Architecture
Natural Language Processing requires the neurons to learn from their predescessor, in a sequential manner. A commonly used model architecture for sequential training is Long Short-Term Memory or LSTM model. LSTM is a type of Recurrent Neural Network.
