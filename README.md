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

## Dependency, Data & Model Initialization
### Dependency Initialization and module downloads
    # Pytorch
    import torch
    import torch.nn as nn
    import pandas as pd
    import numpy as np

    # NLTK
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    # Torchvision
    import torchvision.transforms as transforms
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader, Dataset

    # Collections
    from collections import Counter

    # Regex
    import re

    # AWS Wrangler
    import awswrangler as wr

    nltk.download("punkt")
    nltk.download("stopwords")

### Data Initialization and Preprocessing
    # Create torch seed for reproducibility and fair comparisons
    torch.manual_seeds(10)

    # Initialize a word counter
    word_counter = Counter()

    # Read in the data from the file directory, or cloud storage.
    file_directory = "c:/User/alex/downloads/Reviews.csv" # <- If the data is in the downloads folder.
    df = pd.read_csv(file_directory)

    s3_storage = "s3://foo-bar"
    df = wr.s3.read_csv(s3_storage) ## <- If the data is in the s3 storage. Make sure that you add an appropriate IAM role for SageMaker to read in the data from your s3 bucket.

    df = df[:int(len(df)/10)] ## Limit the rows down to 1/10 of the entire dataset to avoid training overheads. Total of 56845 rows.

    # Text cleaning method
    def text_cleaning(text):
        # Convert the words in lowercase.
        text = text.lower()
        # Remove punctuations and special characters.
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        return text

    # Stop word cleaning method
    def stopword_cleaning(text):
        # Clean the text.
        cleaned_text = text_cleaning(text)
        # Tokenize the cleaned text.
        tokens = word_tokenize(cleaned_text)
        # Remove any stopwords.
        stop_words = set(stopwords.words("english"))
        filtered_tokens = [token for token in tokens if token not in stop_words]

        return filtered_tokens

    # Remove NaN values.
    df["Text"] = df["Text"].fillna("")
    # Ensure all values in "Text" are strings.
    df["Text"] = df["Text"].astype(str)
    # Clean the words and remove any stop words.
    df["tokenize"] = df["Text"].apply(lambda x: stopword_cleaning(x))

    # Indexing and Numericalization
    word_counter = Counter()
    for tokens in df["tokenized"]:
        word_counter.update(tokens)

    vocab = {word: idx + 2 for idx, (word, _) in enumerate(word_counter.most_common())}
    vocab["<PAD>"] = 0  # Padding token
    vocab["<UNK>"] = 1  # Unknown token

    df["numericalized"] = df["tokenized"].apply(lambda x: [vocab.get(token, vocab["<UNK>"]) for token in x])

    # Padding
    max_len = max(map(len, df["numericalized"]))
    df["padded"] = df["numericalized"].apply(lambda x: x + [vocab["<PAD>"]] * (max_len - len(x)))

    df["Score"] = df["Score"] - 1
    X = torch.tensor(df["padded"].tolist())
    y = torch.tensor(df["Score"].tolist())

### Data Loader & Model Architecture
    # Custom Data Loader
    class CustomDataset(Dataset):
    def __init__(self, X, y):
        super(CustomDataset, self).__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]    

    # Model
    class Model(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, bidirectional = True, num_layers = 2, dropout = 0.2):
            super(Model, self).__init__();
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.bidirectional = bidirectional
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
            self.dropout = nn.Dropout(dropout)
            self.batch_norm = nn.BatchNorm1d(hidden_dim * 2 if bidirectional else hidden_dim)
            self.fc1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
        
            # Concatenate the outputs from both directions if bidirectional
            if self.bidirectional:
                lstm_out = torch.cat((lstm_out[:, -1, :self.hidden_dim], lstm_out[:, 0, self.hidden_dim:]), dim=1)
            else:
                lstm_out = lstm_out[:, -1, :]
        
            lstm_out = self.dropout(lstm_out)
            lstm_out = self.batch_norm(lstm_out)
            out = self.fc1(lstm_out)
            out = self.dropout(out)
            out = self.fc2(out)
        
            return out;

    dataset = CustomDataset(X, y)
    n = len(dataset)

    batch_size = 100;
    train_size = int(0.75 * n); # Setting 75% of dataset for training dataset
    validation_size = int(0.15 * n); # 15% of all remaining 25% for validation dataset
    test_size = n - train_size - validation_size; # 10% of remaining for test dataset

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])

    # Create DataLoader for each train, validation, and test datasets.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4);
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Defining hyperparameters
    vocab_size = len(vocab);
    embedding_dim = 100;
    hidden_dim = 128;
    output_dim = 5;

    model = Model(vocab_size, embedding_dim, hidden_dim, output_dim);

    criterion = nn.CrossEntropyLoss();
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001);

    num_epochs = 10;

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}')
        model.train()  # Train the model
        total_loss = 0  # Initialize the Loss to 0

        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            print(f'Processing batch {batch_idx+1}')
            optimizer.zero_grad()  # Clearing out the Gradient Descent

            # Forward pass
            inputs = inputs.to(torch.int64)
            labels = labels.to(torch.int64)
            outputs = model(inputs)

            # Debugging shapes and types
            print(f'Outputs shape: {outputs.shape}, Labels shape: {labels.shape}')
            print(f'Outputs dtype: {outputs.dtype}, Labels dtype: {labels.dtype}')

            # Calculate the loss
            try:
                loss = criterion(outputs, labels)
                print(f'Loss: {loss.item()}')
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f'Error in loss calculation: {e}')
                break  # Exit the loop if there's an error in loss calculation

            total_loss += loss.item();

        # Print average loss for each epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader)}');
    
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










  
  
