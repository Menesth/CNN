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
EPOCHS = 10
DROPOUT = 0.2
LR = 1e-3
BATCH_SIZE = 64
LOSS = nn.CrossEntropyLoss()
################################

####### Data Preparation #######
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
################################

####### Model #######
class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, max_pool_kernel_size=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.maxpool = nn.MaxPool2d(kernel_size=max_pool_kernel_size)

    def forward(self, x):      # [batch_size, in_channels, 28, 28]
        x = self.conv1(x)      # [batch_size, out_channels, 28, 28]
        x = self.batchnorm(x)  # [batch_size, out_channels, 28, 28]
        x = self.relu(x)       # [batch_size, out_channels, 28, 28]
        x = self.dropout(x)    # [batch_size, out_channels, 28, 28]
        x = self.maxpool(x)    # [batch_size, out_channels, 28/max_pool_kernel_size, 28/max_pool_kernel_size]
        return x

class CNN(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=16, out_channels=16, max_pool_kernel_size=2, out_ffw = 2 ** 6):
        super().__init__()
        self.block1 = CNN_block(in_channels=in_channels, out_channels=hidden_channels)
        self.block2 = CNN_block(in_channels=hidden_channels, out_channels=out_channels)
        self.ffw = nn.Linear(out_channels * ((28//(max_pool_kernel_size) ** 2) ** 2), out_ffw)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(out_ffw, 10)

    def forward(self, x):   # [batch_size, in_channels, 28, 28]
        x = self.block1(x)  # [batch_size, out_channels, 28/max_pool_kernel_size, 28/max_pool_kernel_size]
        x = self.block2(x)  # [batch_size, out_channels, 28/max_pool_kernel_size/max_pool_kernel_size, 28/max_pool_kernel_size/max_pool_kernel_size]
        b, c, h, w = x.shape
        x = x.view(b, c*h*w) # [batch_size, out_channels * 28/max_pool_kernel_size/max_pool_kernel_size * 28/max_pool_kernel_size/max_pool_kernel_size]
        x = self.ffw(x)      # [batch_size, out_ffw]
        x = self.relu(x)     # [batch_size, out_ffw]
        x = self.dropout(x)  # [batch_size, out_ffw]
        x = self.fc(x)       # [batch_size, 10]
        return x

    def stochastic_predict(self, x):
        logits = self.forward(x)                       # [batch_size, 10]
        probs = F.softmax(logits, dim=-1)              # [batch_size, 10]
        pred = torch.multinomial(probs, num_samples=1) # [batch_size, 1]
        return pred.squeeze()                          # [batch_size]

    def argmax_predict(self, x):
        logits = self.forward(x)            # [batch_size, 10]
        probs = F.softmax(logits, dim=-1)   # [batch_size, 10]
        pred = torch.argmax(probs, dim=-1)  # [batch_size]
        return pred
############################

##### Initialize model ######
model = CNN().to(device)

nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'number of parameters: {nb_params}')
############################

####### Training ########
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 if epoch < 6 else 0.1)

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

    scheduler.step()
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
