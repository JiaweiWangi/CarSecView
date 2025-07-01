import torch
from torch import nn

def init_model():   
    hidden_size = 50
    input_size = 1
    model_file = 'model_original.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(input_size, hidden_size, 2)
    model.load_state_dict(torch.load(f'./model/{model_file}', map_location=device))
    model.to(device)
    model.eval()
    return model,device

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通过编码器
        out, _ = self.encoder(x)
        # 取最后一个时间步的输出
        out = out[:, -1]
        out = self.decoder(out)
        # out = self.sigmoid(out)
        return out.unsqueeze(-1)

# 训练函数
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    model.to(device)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        batch_count = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        avg_train_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.8f}")
    return model
