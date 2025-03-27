import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import trange

####### Hyperparameters #######
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
EPOCHS = 5
DROPOUT = 0.2
LR = 1e-3
BATCH_SIZE = 64
LOSS = nn.CrossEntropyLoss()
################################

####### Data Preparation #######
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
test_data = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
################################


####### Model #######
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):            # [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))    # [batch_size, 16, 28, 28]
        x = F.max_pool2d(x, 2)       # [batch_size, 16, 14, 14]
        x = F.relu(self.conv2(x))    # [batch_size, 32, 14, 14]
        x = F.max_pool2d(x, 2)       # [batch_size, 32, 7, 7]
        x = x.view(x.size(0), -1)    # [batch_size, 32*7*7]

        x = self.dropout(F.relu(self.fc1(x)))  # [batch_size, 128]
        x = self.fc2(x)                        # [batch_size, 10]
        return x
    
    def stochastic_predict(self, x):
        logits = self.forward(x)                       # [batch_size, 10]
        probs = F.softmax(logits, dim=-1)              # [batch_size, 10]
        pred = torch.multinomial(probs, num_samples=1) # [batch_size, 1]
        return pred.squeeze()                          # [batch_size]

    def argmax_predict(self, x):
        logits = self.forward(x)            # [batch_size, 10]
        pred = torch.argmax(logits, dim=-1) # [batch_size]
        return pred
############################

##### Initialize model ######
model = CNN().to(device)

nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'number of parameters: {nb_params}')
############################

####### Training ########
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in trange(EPOCHS):
    model.train()
    running_loss = 0.0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        
        y_hat = model(X)
        loss = LOSS(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"epoch [{epoch+1}/{EPOCHS}], batch loss: {avg_train_loss:.4f}")
############################

####### Testing ########
model.eval()
sto_correct, argmax_correct = 0, 0
total = 0

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        sto_pred = model.stochastic_predict(X)
        argmax_pred = model.argmax_predict(X)
        total += y.size(0)
        sto_correct += (sto_pred == y).sum().item()
        argmax_correct += (argmax_pred == y).sum().item()

sto_accuracy = 100 * sto_correct / total
argmax_accuracy = 100 * argmax_correct / total
print(f"Test accuracy (with stochastic prediction): {sto_accuracy:.2f}%")
print(f"Test accuracy (with argmax prediction): {argmax_accuracy:.2f}%")
############################
