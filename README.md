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

## Initial Model Training Approach
### Hyperparameters
    Batch Size: 100
    Input Dimension: 128
    Output Dimension: 5
    Training Size: 75%

### Optimizer Parameter
    Optimizer: Adam
    Learning Rate: 0.001

### Training Output

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
  
    Processing batch 85
    Outputs shape: torch.Size([40, 5]), Labels shape: torch.Size([40])
    Outputs dtype: torch.float32, Labels dtype: torch.int64
    Loss: 1.2020289897918701
  
    Processing batch 86
    Outputs shape: torch.Size([40, 5]), Labels shape: torch.Size([40])
    Outputs dtype: torch.float32, Labels dtype: torch.int64
    Loss: 1.2219252586364746

### Result Interpretations & Adjustments
Initially created model was producing poor results. Loss was too high, and further training after 7 epochs was sufficient to recognize the need for changing model parameters. There are various factors causing high loss during the training, but the only way to lower the loss and increase the model accuracy was a consistent trial and error process.
One of the most common way to lower the traning loss is to lower the learning rate, achieving the following model adjustments:
- Convergence Stability: As training progresses, the model parameters should ideally converge to a set of values that minimize the loss function. A high learning rate can cause the optimizer to overshoot the optimal values, leading to oscillations or even divergence. Lowering the learning rate helps in fine-tuning the parameters, ensuring more stable convergence.
- Escape Local Minima: In the initial stages of training, a higher learning rate can help in escaping local minima (suboptimal points) and exploring the parameter space more effectively. As training progresses, lowering the learning rate can help in settling into a global or better local minimum.
- Learning Rate Scheduling: Often, a learning rate schedule is used, which gradually decreases the learning rate over time. Techniques like step decay, exponential decay, or adaptive learning rate methods (like ReduceLROnPlateau) are employed to adjust the learning rate as training progresses.

## Tuned Model Training Approach
### Hyperparameters
    Batch Size: 50
    Input Dimension: 128
    Output Dimension: 5
    Training Size: 75%

### Optimizer Parameter
    Optimizer: Adam
    Learning Rate: 0.0001

### Training Output



### Training Loss Plot


### Test & Validation Output


### Test & Validation Loss Plot

    
### Result Interpretations & Adjustments
It was surprising to see how the model accuracy and loss got worse than before. The model outputs directly contradicted to the common practices of improving the model by lowering the learning rate. Although uncommon, such was likely due to the following reasons:

1. Stuck in Local Minima or Plateaus: If the training process gets stuck in a local minimum or a plateau (a region where the loss function has very small gradients), increasing the learning rate can help the optimizer to escape these regions. This can allow the model to explore more of the parameter space and potentially find a better path to the global minimum.

2. Adaptive Learning Rate Methods: Some adaptive learning rate algorithms, such as Adam, RMSprop, and Adagrad, adjust the learning rate based on the magnitude of the gradients. In these methods, the effective learning rate can increase in some dimensions while decreasing in others. These adjustments can help improve convergence speed and performance.

3. Cyclic Learning Rates: Cyclic learning rate schedules, such as the one used in Cyclical Learning Rates (CLR) and the One Cycle Policy, involve periodically increasing and then decreasing the learning rate. This technique can sometimes lead to improved performance and faster convergence. The idea is to allow the learning rate to oscillate between a lower and an upper bound, encouraging the optimizer to escape local minima and explore different regions of the loss landscape.

4. Learning Rate Warm-up: In some training regimes, particularly with very deep neural networks or transformers, it's beneficial to start with a very low learning rate and gradually increase it (warm-up) for the initial few epochs. This helps in stabilizing the training process early on. After the warm-up phase, the learning rate is usually lowered gradually.

New strategy to improve the model's accuracy was to increase the learning rate instead, and involving some dropouts and additional layers to the model. Multiples of training sessions were conducted alongside of tweaking the model architecture, and the following was achieved.

## Fine-tuned Model Training Approach
### Hyperparameters
Batch Size: 50
Input Dimension: 128
Output Dimension: 5
Training Size: 75%

### Optimizer Parameter
Optimizer: Adam
Learning Rate: 0.01

### Training Output
    Starting epoch 1

### Training Loss Plot
![Test & Validation Loss Plot](https://github.com/dntjdgur/ml-nlp/blob/main/images/init_training_loss.png)

### Test & Validation Output

    Validation Loss: 1.5438, Validation Accuracy: 0.6000
    Test Loss: 1.6079, Test Accuracy: 0.5826

### Test & Validation Loss Plot
![Test & Validation Loss Plot](https://github.com/dntjdgur/ml-nlp/blob/main/images/init_test_val_loss.png)
