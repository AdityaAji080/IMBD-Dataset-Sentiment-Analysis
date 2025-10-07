import random
import time
from torchtext import data
from torchtext import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F

#Assigning a random seed for reproducibility
seed = 420

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True #For CUDA GPU only

#Define field: tokenizer, padding
Text = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  include_lengths = True,
                  pad_first = True)

#Setting to float for binary classification
Label = data.LabelField(dtype = torch.float)

#Split dataset into training and test sets
df_train, df_test = datasets.IMDB.splits(Text, Label)

#Split training dataset into training and validation sets
df_train, df_valid = df_train.split(random_state = random.seed(seed))

#Build a vocabulary
Text.build_vocab(df_train,
                 max_size = 25_000)

Label.build_vocab(df_train)

pad_idx = Text.vocab.stoi[Text.pad_token]

#Check for CUDA GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#Define iterators
train_iter, valid_iter, test_iter = data.BucketIterator.splits((df_train, df_valid, df_test),
                                                               batch_size = 64,
                                                               sort_within_batch = True,
                                                               sort_key = lambda x: len(x.text),
                                                               device = device)

#Define the RNN model
class CNN(nn.Module):
    def __init__(self, input_size, embedding_dim, output_size, pad_idx, dropout = 0.5, kernel_sizes = (1,2,3), num_feature_maps = 1):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx = pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=num_feature_maps,
                      kernel_size=k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_feature_maps * len(kernel_sizes), output_size)

        with torch.no_grad():
            self.embedding.weight[pad_idx].fill_(0)

    def forward(self, text, text_length):
        embedded = self.dropout(self.embedding(text))
        x = embedded.permute(1, 2, 0)
        feats = []
        for conv in self.convs:
            c = F.relu(conv(x))  # [batch, out_ch=1, L']
            p = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)  # [batch, 1]
            feats.append(p)

        cat = self.dropout(torch.cat(feats, dim=1))

        out = self.fc(cat)
        return out

INPUT_DIM = len(Text.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1

#Instance of RNN
model = CNN(input_size = INPUT_DIM, embedding_dim = EMBEDDING_DIM, output_size = OUTPUT_DIM, pad_idx = pad_idx, dropout = 0.5, kernel_sizes=(1,2,3), num_feature_maps = 1).to(device)

#Define the optimizers
optimizer_SGD = optim.SGD(model.parameters(), lr = 0.001)
optimizer_Adam = optim.Adam(model.parameters(), lr = 0.001)
optimizer_Adagrad = optim.Adagrad(model.parameters(), lr = 0.001)

#Binary Classification
criterion = nn.BCEWithLogitsLoss().to(device)

#Model Training
def binary_accuracy(preds, y):
    rounded_prediction = torch.round(torch.sigmoid(preds))
    correct = (rounded_prediction == y).float()
    return correct.sum() / len(correct)

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_accuracy = 0
    model.train()

    for batch in iterator:

        optimizer.zero_grad()
        text, text_length = batch.text
        predictions = model(text,text_length).squeeze(1)
        loss = criterion(predictions, batch.label)
        accuracy = binary_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()

    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_accuracy = 0

    model.eval()
    with torch.no_grad():

        for batch in iterator:
            text, text_length = batch.text
            predictions = model(text,text_length).squeeze(1)
            loss = criterion(predictions, batch.label)
            accuracy = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)

data = []

N_EPOCHS = 50

for epoch in range(N_EPOCHS):
    start = time.time()
    train_loss, train_accuracy = train(model, train_iter, optimizer_Adam, criterion)
    valid_loss, valid_accuracy = evaluate(model, valid_iter, criterion)
    end = time.time()

    # print(f'Epoch {epoch+1}:')
    # print(f'  Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy*100:.2f}%')
    # print(f'  Validation Loss: {valid_loss:.3f}, Validation Accuracy: {valid_accuracy*100:.2f}%')
    # print(f'  Time elapsed: {end-start:.2f}s')

    data.append({
        "Epoch": [epoch + 1],
        "Train Loss": [round(train_loss, 3)],
        "Train Accuracy (%)": [round(train_accuracy * 100, 2)],
        "Validation Loss": [round(valid_loss, 3)],
        "Validation Accuracy (%)": [round(valid_accuracy * 100, 2)],
        "Time (s)": [round(end - start, 2)]
    })
df = pd.DataFrame(data)
print(df.to_string(index=False))