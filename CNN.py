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
DROPOUT = 0.1
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
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = F.dropout(out, p=DROPOUT)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual # residual connection
        out = F.relu(out, inplace=True)
        out = F.dropout(out, p=DROPOUT)
        return out

class CNN(nn.Module):
    def __init__(self, layers=[2, 2, 2], channels = [32, 64, 128], num_classes=10):
        super().__init__()
        self.in_channels = channels[0]
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(self.in_channels)

        strides = [1] + [2 for _ in range(len(layers) - 1)]
        self.layers = nn.Sequential(*[self._make_layer(out_channels=channels[i], blocks=layers[0], stride = strides[i]) for i in range(len(layers))])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):                        # [batch_size, 1, 28, 28]
        out = self.conv(x)                       # [batch_size, 64, 14, 14]
        out = F.relu(self.bn(out), inplace=True) # [batch_size, channels[0], 14, 14]
        out = F.dropout(out, p=DROPOUT)          # [batch_size, channels[0], 14, 14]

        out = self.layers(out)                   # [batch_size, channels[-1], 4, 4]

        out = self.avgpool(out)                  # [batch_size, channels[-1], 1, 1]

        b, c, h, w = out.shape                      
        out = out.view(b, c * h * w)             # [batch_size, channels[-1]]
        out = self.fc(out)                       # [batch_size, num_classes]
        return out

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
                )

        layers = []
        layers.append(ResBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(ResBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

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
