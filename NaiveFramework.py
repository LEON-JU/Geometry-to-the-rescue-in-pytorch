import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable

# 全局超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# 定义简单的Autoencoder模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 加载Kitti数据集
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

kitti_dataset = ImageFolder(root='path/to/kitti/dataset', transform=transform)
kitti_loader = DataLoader(dataset=kitti_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 初始化模型、损失函数和优化器
autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for data in kitti_loader:
        img, _ = data
        img = Variable(img)

        # 前向传播
        output = autoencoder(img)
        loss = criterion(output, img)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 打印每个epoch的损失
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
test_image = Variable(kitti_dataset[0][0].unsqueeze(0))
reconstructed_image = autoencoder(test_image)
# 在这里进行进一步的测试和可视化

# 保存模型
torch.save(autoencoder.state_dict(), 'autoencoder_model.pth')