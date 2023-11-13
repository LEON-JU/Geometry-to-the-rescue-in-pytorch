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

'''
Autoencoder
Input size: 188 x 620
'''
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.L1 = nn.Sequential(
            # C1
            nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride = 4, padding = 0),
            nn.ReLU()
        )

        self.L2 = nn.Sequential(
            # P1
            nn.MaxPool2d(kernel_size = 3, stride = 2), 
            # C2
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU()
        )

        self.L3 = nn.Sequential(
            # P2
            nn.MaxPool2d(3, 2),
            # C3
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU()
        )

        self.L4 = nn.Sequential(
            # C4
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU()
        )

        self.L5 = nn.Sequential(
            # C5
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU()
        )

        self.L6 = nn.Sequential(
            # C6
            # TODO: 修改stride和paddings
            nn.Conv2d(256, 2048, 5, stride = 0, padding=2),
            nn.ReLU()
        )


        
        # TODO: 实现FCN层

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
