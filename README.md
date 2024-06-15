# Sentiment Natural Language Processing Model
## Project Purpose
This project is to build a basic NLP model that can perform the sentiment analysis. Sentiment analysis, also known as opinion mining or emotion AI, uses the NLP model to identify and quantify the human emotions portraited from the given texts. This model is expected to provide quantified emotion level scales from 1 through 5, 1 being the most dissatisfied state, and 5 being the most satisfied state.
## Data Source
Data source was obtained from Kaggle, titled by [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
## Descriptive Statistics
Data is composed of 568,454 rows and 10 columns. Column "Text" contains the review texts, and "Score" contains the quantified scale of satisfaction from food, ranging from 1 through 5.
## Model Architecture
Natural Language Processing requires the neurons to learn from their predescessor, in a sequential manner. One of the most common neural network being used to analyze and predict the output is RNN. RNN (Recurrent Neural Network) is a type of neural network architecture in which the output of a neuron is fed back as an input to the network at the next time step, retaining memory of past inputs. Although it may sound RNN to be overly complex, it is actually consisted of a simple structure. And because of its simple structure, RNN is computationally less intensive. RNN, however, constantly suffers from vanishing and exploding gradient problems, which ultimately affects in learning long-range dependencies in sequences and difficulties in establishing the adequate layers.

LSTM (Long Short-Term Memory) model is a type of RNN that is more complex in its composition. LSTM is designed to overcome the limitations of traditional RNNs in learning and remembering long-range dependencies in sequential data. Each LSTM unit consists of the forget gate, the input gate, and the output gate. The forget gate determines which information from the previous state should be disregarded, the input gate decides which new information should be added to the current cell state, and the output gate determines what information from the current cell state should be outputted. The entire LSTM network is consisted of multiple LSTM units, allowing each unit to learn from its predecessor and improving the accuracy of the output. LSTM is a great fit for analyzing time-series or sequential datasets, commonly applied for Natural Language Processings. In this project, we will be applying LSTM as a basis model architecture to predict the sentiment of given input texts.

## Learning Environment
### AWS SageMaker
    Instance: ml.t3.2xlarge
    Storage: 100 GB
    Image: SageMaker Distribution 1.8
    Lifecycle Configuration: None

    
Output:

    Starting epoch 1
    Processing batch 1
    Outputs shape: torch.Size([100, 5]), Labels shape: torch.Size([100])
    Outputs dtype: torch.float32, Labels dtype: torch.int64
    Loss: 1.735997200012207

    Processing batch 2
    Outputs shape: torch.Size([100, 5]), Labels shape: torch.Size([100])
    Outputs dtype: torch.float32, Labels dtype: torch.int64
    Loss: 1.7038110494613647
  
    ...
  
    Processing batch 709
    Outputs shape: torch.Size([40, 5]), Labels shape: torch.Size([40])
    Outputs dtype: torch.float32, Labels dtype: torch.int64
    Loss: 1.2020289897918701
  
    Processing batch 710
    Outputs shape: torch.Size([40, 5]), Labels shape: torch.Size([40])
    Outputs dtype: torch.float32, Labels dtype: torch.int64
    Loss: 1.2219252586364746

At this point, another epoch training session was not made and the kernel was immediately interrupted since the loss was not improved.










  
  
