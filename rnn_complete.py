import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import re
import copy


# ===================== Dataset =====================
class CharDataset(Dataset):
    def __init__(self, data, sequence_length, stride, vocab_size):
        self.data = data
        self.sequence_length = sequence_length
        self.stride = stride
        self.vocab_size = vocab_size
        self.sequences = []
        self.targets = []
        
        # Create overlapping sequences with stride
        for i in range(0, len(data) - sequence_length, stride):
            self.sequences.append(data[i:i + sequence_length])
            self.targets.append(data[i + 1:i + sequence_length + 1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return sequence, target

# ===================== Model =====================
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim=30):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_dim)
        # TODO: Initialize your model parameters as needed e.g. W_e, W_h, etc.

        # 1) RNN parameters
        self.W_xh = nn.Parameter(torch.randn(embedding_dim, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h  = nn.Parameter(torch.zeros(hidden_size))

        # 2) output projection  
        self.W_hy = nn.Linear(hidden_size, output_size)
        self.b_y  = nn.Parameter(torch.zeros(output_size))
    

    def forward(self, x, hidden):
        """
        x in [b, l] # b is batch_size and l is sequence length
        """

        x_embed = self.embedding(x)  # [b=batch_size, l=sequence_length, e=embedding_dim]
        b, l, _ = x_embed.size()
        x_embed = x_embed.transpose(0, 1) # [l, b, e]

        if hidden is None:
            h_t_minus_1 = self.init_hidden(b)
        else:
            h_t_minus_1 = hidden


        
        output = []
        for t in range(l):                            
            # RNN step: h_t = tanh( x_t @ W_xh^T + h_prev @ W_hh^T + b_h )
            x_t = x_embed[t]
            h_t = torch.tanh(( x_t @ self.W_xh) + (h_t_minus_1@self.W_hh) + self.b_h)

            output.append(h_t)
            h_t_minus_1 = h_t
            
            #  TODO: Implement forward pass for a single RNN timestamp, append the hidden to the output 


        output = torch.stack(output) # [l, b, e]
        output = output.transpose(0, 1) # [b, l, e]
        
        # TODO set these values after completing the loop above

        final_hidden = h_t.clone()  # [b, h] 
        logits = self.W_hy(output) + self.b_y # [b, l, vocab_size=v] 
        return logits, final_hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

# ===================== Training =====================
device = 'cpu'
print(f"Using device: {device}")

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = re.sub(r'[^a-z.,!?;:()\[\] ]+', '', text)
    return text

# To debug your model you should start with a simple sequence an RNN should predict this perfectly#
sequence = "abcdefghijklmnopqrstuvwxyz" * 100
#sequence = read_file("warandpeace.txt") # Uncomment to read from file
vocab = sorted(set(sequence))


char_to_idx = {char: i for i, char in enumerate(vocab)}
idx_to_char = {i: char for i, char in enumerate(vocab)}




    
data = [char_to_idx[char] for char in sequence]

# TODO Understand and adjust the hyperparameters once the code is running
sequence_length = 10 # Length of each input sequence
stride = 1           # Stride for creating sequences
embedding_dim = 64      # Dimension of character embeddings
hidden_size = 256        # Number of features in the hidden state of the RNN
learning_rate = 0.0005    # Learning rate for the optimizer
num_epochs = 3      # Number of epochs to train
batch_size = 128        # Batch size for training
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

model = CharRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)

#TODO: Split the data into 90:10 ratio with PyTorch indexing
train_data = data_tensor[:train_size]
test_data = data_tensor[train_size:]

"""
print("data:", data)
print("data_tensor:", data_tensor)
print("train_data:", train_data)
print("test_data:", test_data)
print(len(train_data))
print(len(test_data))
"""


train_dataset = CharDataset(train_data, sequence_length, stride, output_size)
test_dataset = CharDataset(test_data, sequence_length, stride, output_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    hidden = None

    for batch_inputs, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        batch_inputs = batch_inputs.to(torch.long)

        output, hidden = model(batch_inputs, hidden)
        # Detach removes the hidden state from the computational graph.
        # This is prevents backpropagating through the full history, and
        # is important for stability in training RNNs
        hidden = hidden.detach()

        b, l, v = output.size()

        logits = output.view(b * l, v)
        labels = batch_targets.view(b * l)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # TODO compute the loss, backpropagate gradients, and update total_loss Done


print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")


def test_model(model, test_loader, criterion, device):
    """
    Evaluate the CharRNN model on a test dataset.
    Prints average test loss and per‐character accuracy.
    """
    if len(test_loader) == 0:
        print("Test set is empty. No evaluation performed.")
        return

    model.eval()
    total_loss = 0.0
    total_chars = 0
    correct_chars = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            logits, _ = model(inputs, None)
            b, l, vocab_size = logits.size()
            
            # flatten for loss/accuracy
            logits_flat = logits.view(b * l, vocab_size)
            labels_flat = targets.view(b * l)
            
            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item() * labels_flat.size(0)
            
            preds = torch.argmax(logits_flat, dim=1)
            correct_chars += (preds == labels_flat).sum().item()
            total_chars += labels_flat.size(0)

    if total_chars == 0:
        print("No characters processed during testing. Check the test dataset.")
        return

    avg_loss = total_loss / total_chars
    accuracy = correct_chars / total_chars
    print(f"Test Loss: {avg_loss:.4f}, Character Accuracy: {accuracy:.2%}")

    model.train()  # switch back to train mode

test_model(model, test_loader, criterion, device)




# Test the model
# TODO: Implement a test loop to evaluate the model on the test set

# ===================== Text Generation =====================
def sample_from_output(logits, temperature=1.0):
    """
    Sample from the logits with temperature scaling.
    logits: Tensor of shape [batch_size, vocab_size] (raw scores, before softmax)
    temperature: a float controlling the randomness (higher = more random)
    """
    if temperature <= 0:
        temperature = 0.00000001
    # Apply temperature scaling to logits (increase randomness with higher values)
    scaled_logits = logits / temperature 
    # Apply softmax to convert logits to probabilities
    print(scaled_logits.size())

    probabilities = F.softmax(scaled_logits, dim=0) 
    print(probabilities)
    
    # Sample from the probability distribution
    sampled_idx = torch.multinomial(probabilities, 1)
    return sampled_idx

def generate_text(model, start_text, n, k, temperature=1.0):
    """
        model: The trained RNN model used for character prediction.
        start_text: The initial string of length `n` provided by the user to start the generation.
        n: The length of the initial input sequence.
        k: The number of additional characters to generate.
        temperature: Optional
        A scaling factor for randomness in predictions. Higher values (e.g., >1) make 
            predictions more random, while lower values (e.g., <1) make predictions more deterministic.
            Default is 1.0.
    """
    model.eval()
    hidden = None
    start_text = start_text.lower()

    convert = []
    for i in start_text:
        convert.append(char_to_idx[i])

    
    for j in range(k):

        logits = torch.tensor(convert[-n:]).unsqueeze(0)
        input_tensor = torch.tensor(logits, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, hidden = model(logits, hidden)
        logits = logits[0,-1]
            

        result = (sample_from_output(logits,temperature)).item()
        next_char = idx_to_char[result]
        convert.append(result)
        start_text += next_char
    



    #TODO: Implement the rest of the generate_text function
    # Hint: you will call sample_from_output() to sample a character from the logits

    return start_text

print("Training complete. Now you can generate text.")
while True:
    start_text = input("Enter the initial text (n characters, or 'exit' to quit): ")
    
    if start_text.lower() == 'exit':
        print("Exiting...")
        break
    
    n = len(start_text) 
    k = int(input("Enter the number of characters to generate: "))
    temperature_input = input("Enter the temperature value (1.0 is default, >1 is more random): ")
    temperature = float(temperature_input) if temperature_input else 1.0
    
    completed_text = generate_text(model, start_text, n, k, temperature)
    
    print(f"Generated text: {completed_text}")