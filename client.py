import os
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 打印当前使用的 GPU 编号
gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not specified')
print(f"Using GPU ID: {gpu_id}")

# 选择可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义一个简单的全连接神经网络模型
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, input_size)  # 展平输入
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 定义 Flower 客户端
class SimpleClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

   
    
    def fit(self, parameters, config):
       self.set_parameters(parameters)
       criterion = nn.CrossEntropyLoss()
       optimizer = optim.SGD(self.model.parameters(), lr=0.001)
       num_examples = len(self.train_loader.dataset)
    
    # 添加训练准确率跟踪
       correct = 0
       total = 0
    
       self.model.train()
       for epoch in range(1):  # 训练 1 个 epoch
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 返回训练准确率
       accuracy = correct / total
       return self.get_parameters(config={}), num_examples, {"accuracy": accuracy}  # 添加指标

    def evaluate(self, parameters, config):
        print("evaluate执行中")
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        num_examples = len(self.test_loader.dataset)
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        loss = total_loss / len(self.test_loader)
        accuracy = float(correct / num_examples)  # 确保 accuracy 是 float 类型

        # 打印准确率和总测试样本数
        print(f"Client accuracy: {accuracy}, num_examples: {num_examples}")
        return float(loss), num_examples, {"accuracy": accuracy}

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 定义数据路径
data_path = './data'

# 加载训练集，download=True 表示如果数据不存在则自动下载
train_dataset = datasets.MNIST(root=data_path, train=True,
                               download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# 加载测试集，download=True 表示如果数据不存在则自动下载
test_dataset = datasets.MNIST(root=data_path, train=False,
                              download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# 初始化模型
input_size = 28 * 28  # MNIST 图像大小为 28x28
hidden_size = 20
num_classes = 10
model = SimpleNet(input_size, hidden_size, num_classes)

# 启动 Flower 客户端
client = SimpleClient(model, train_loader, test_loader)
fl.client.start_client(server_address="0.0.0.0:8080", client=client)  # 使用 start_client() 替代 start_numpy_client()

print("启动客户端")