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
    Batch Size: 150
    Input Dimension: 128
    Output Dimension: 5
    Training Size: 75%

### Optimizer Parameter
    Optimizer: Adam
    Learning Rate: 0.001

### Training Output
    Starting epoch 1
    Batch: 1 Loss: 1.666138768196106
    Batch: 2 Loss: 1.7109413146972656
    Batch: 3 Loss: 1.6612130403518677
    Batch: 4 Loss: 1.6497366428375244
    Batch: 5 Loss: 1.7631101608276367
    ...
    Batch: 166 Loss: 0.2060454934835434
    Batch: 167 Loss: 0.2550095319747925
    Batch: 168 Loss: 0.3515404760837555
    Batch: 169 Loss: 0.21221043169498444
    Batch: 170 Loss: 0.34858688712120056
    Batch: 171 Loss: 0.44689175486564636
    Epoch 10/10, Loss: 0.3143108763367112

### Training Loss Plot
![Test & Validation Loss Plot](https://github.com/dntjdgur/ml-nlp/blob/main/images/init_training_loss.png)

### Validation & Test Output
    Validation Loss: 2.0131, Validation Accuracy: 0.5812
    Test Loss: 1.8693, Test Accuracy: 0.6046

### Test & Validation Loss Plot
![Validation & Test Loss Plot](https://github.com/dntjdgur/ml-nlp/blob/main/images/init_val_test_loss.png)

### Result Interpretations & Adjustments
Initial model output indicated a good start in the analysis. Loss was somewhere in between 0.2 to 0.4 range, and gradually decreased over several epochs in the training.
But the model was not producing similar stability in the validation and test stages. As the plot indicates, the loss value massively fluctuated between maximum 2.5 to minimum 0.5. A new strategy to lower the loss needs to be taken as the next step in the tuning process.
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
    Learning Rate: 0.001

### Training Output
    Starting epoch 1
    Batch: 1 Loss: 1.6897330284118652
    Batch: 2 Loss: 1.7010935544967651
    Batch: 3 Loss: 1.6997413635253906
    Batch: 4 Loss: 1.7000572681427002
    Batch: 5 Loss: 1.6674854755401611
    ...
    Batch: 210 Loss: 0.10914823412895203
    Batch: 211 Loss: 0.12155231833457947
    Batch: 212 Loss: 0.09095839411020279
    Batch: 213 Loss: 0.09806714206933975
    Batch: 214 Loss: 0.0602705255150795
    Epoch 25/25, Loss: 0.10540465810379693

### Training Loss Plot
![Training Loss Plot](https://github.com/dntjdgur/ml-nlp/blob/main/images/tuned_training_loss.png)

### Test & Validation Output
    Validation Loss: 2.6043, Validation Accuracy: 0.6079
    Test Loss: 2.5778, Test Accuracy: 0.6104

### Test & Validation Loss Plot
![Validation & Test Loss Plot](https://github.com/dntjdgur/ml-nlp/blob/main/images/tuned_val_test_loss.png)
    
### Result Interpretations & Adjustments
This time, the batch size was increased to 200, training through 100 epochs, and the remaining parameters stayed the same. The reason for this drastic change in the batch size is to observe the changes in the model loss and prediction accuracy, and if the model begin to show improvements, then the changes in the batch size may suffice the tuning procedures. As the training progressed, loss was dramatically decreased down to 0.02 - 0.2 range, which sounded to be a good sign. This, however, could also be the indication of overfitting as the number of samples were reduced down to 1/50 of the entire dataset (total of 560,000 rows). Depending on the validation and test runs, it can be concluded whether the model is indeed working well or overfitting.

Total of 25 epochs, training resulted in a relatively acceptable loss rate. However, the validation and test indicate that the accuracy did not get any better. This would be an indication that the model was overfitting after some trials, and could mean that the model needs further adjustments.

What seems obvious is that better and more meticulous preprocessing is required. Currently the model analyzes the correlations between the scores and the tokenized words in the review summary, and are mostly concerned with adjectives. But the result of the validation and testing may indicate that adverbs and nouns should also be considered in the analysis as solely performing the analysis on the adjectives can't be enough.

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
![Test & Validation Loss Plot]

### Test & Validation Output

    Validation Loss: 1.5438, Validation Accuracy: 0.6000
    Test Loss: 1.6079, Test Accuracy: 0.5826

### Test & Validation Loss Plot
![Test & Validation Loss Plot]
