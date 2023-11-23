import torch 
import torch.nn as nn
# 测试一下输出尺寸
 
# With square kernels and equal stride
m = nn.Conv2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
 
print(m.weight.data.size())
input = torch.randn(20, 16, 50, 100)
output = m(input)
print(output.shape)