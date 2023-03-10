import torch
import torch.nn as nn
import einops
from tqdm import tqdm

from torch.optim import optimizer

class PatchEmbedding(nn.Module):
  def __init__(self, input_map, kernel_size, channel_size):
    super().__init__()

    self.conv = nn.Conv2d(input_map[0], channel_size, kernel_size, kernel_size, 0)
  
  def forward(self, x):
    x = self.conv(x)

    return x

class LocalMSA(nn.Module):
  def __init__(self, channel_size, num_head):
    super().__init__()

    self.num_head = num_head
    self.scale = (channel_size)**(0.5)

    self.qkv = nn.Linear(channel_size, 3*channel_size)
  
  def forward(self, x):
    x = einops.rearrange(self.qkv(x), 'b g l (d h c) -> d b g h l c', h = self.num_head, d = 3)
    q = x[0]
    k = torch.transpose(x[1], -1, -2)
    v = x[2]
    
    score = torch.matmul(q,k)/self.scale
    score = nn.functional.softmax(score, dim = -1)
    score = torch.matmul(score,v)
    score = einops.rearrange(score, 'b g h l c -> b g l (h c)')

    return score

class DilatedMSA(nn.Module):
  def __init__(self, channel_size, num_head):
    super().__init__()

    self.num_head = num_head
    self.scale = (channel_size)**(0.5)

    self.qkv = nn.Linear(channel_size, 3*channel_size)

  def forward(self, x):
    x = einops.rearrange(self.qkv(x), 'b l g (d h c) -> d b l h g c', h = self.num_head, d = 3)
    q = x[0]
    k = torch.transpose(x[1], -1, -2)
    v = x[2]
    
    score = torch.matmul(q,k)/self.scale
    score = nn.functional.softmax(score, dim = -1)
    score = torch.matmul(score,v)
    score = einops.rearrange(score, 'b l h g c -> b l g (h c)')

    return score

class MLP(nn.Module):
  def __init__(self, channel_size):
    super().__init__()

    self.MLP = nn.Sequential(
        nn.Linear(channel_size, 4*channel_size),
        nn.GELU(),
        nn.Linear(4*channel_size, channel_size)
    )
  
  def forward(self, x):
    x = self.MLP(x)

    return x

class LSTM(nn.Module):
  def __init__(self, channel_size):
    super().__init__()

    self.layer_x = nn.Conv2d(channel_size, 4*channel_size, 1, 1, 0, bias = False)
    self.layer_h = nn.Conv2d(channel_size, 4*channel_size, 1, 1, 0)

    self.softmax2d = nn.Softmax2d()
  
  def forward(self, x, hidden):
    h, C = hidden
    gates = self.layer_x(x) + self.layer_h(h)
    forget, input, state, output = gates.chunk(4, dim = 1)

    forget = self.softmax2d(forget)
    input = self.softmax2d(input)
    state = torch.tanh(state)
    output = self.softmax2d(output)

    C = forget*C + input*state
    h = output*torch.tanh(C)

    return h, (h, C)

class LDAttentionBlock(nn.Module):
  def __init__(self, local_patch_num, channel_size, num_head):
    super().__init__()

    self.local_patch_num = local_patch_num

    self.Conv = nn.Conv2d(channel_size, channel_size, 2, 2, 0)
    self.LocalMSA = LocalMSA(channel_size, num_head)
    self.MLP_1 = MLP(channel_size)
    self.DilatedMSA = DilatedMSA(channel_size, num_head)
    self.MLP_2 = MLP(channel_size)
    self.LSTM = LSTM(channel_size)
  
    self.initial = 0

  def forward(self, x, hidden):
    x = self.Conv(x)
    global_patch_num_x = int(x.shape[-1]/self.local_patch_num)
    x = Window_partition(x, self.local_patch_num)
    x = x + self.LocalMSA(x)
    x = x + self.MLP_1(x)
    x = einops.rearrange(x, 'b g l c -> b l g c')
    x = x + self.DilatedMSA(x)
    x = x + self.MLP_2(x)
    x = Window_reverse(x, global_patch_num_x, self.local_patch_num)
    temp = list()
    for i in x:
      t, hidden = self.LSTM(i.unsqueeze(dim = 0), hidden)
      temp.append(t)
    x = torch.cat(temp, dim = 0)

    return x, hidden

class Modules(nn.Module):
  def __init__(self, input_map, kernel_size, local_patch_num, channel_size, num_head):
    super().__init__()

    self.local_patch_num = local_patch_num
    
    self.Embedding = PatchEmbedding(input_map, kernel_size, channel_size)
    self.Stage_1 = LDAttentionBlock(local_patch_num, channel_size, num_head)
    self.Stage_2 = LDAttentionBlock(local_patch_num, channel_size, num_head)
    self.Stage_3 = LDAttentionBlock(local_patch_num, channel_size, num_head)
    self.Stage_4 = LDAttentionBlock(local_patch_num, channel_size, num_head)
  
  def forward(self, x, h1, h2, h3, h4):
    x = self.Embedding(x)
    x, h1 = self.Stage_1(x, h1)
    x, h2 = self.Stage_2(x, h2)
    x, h3 = self.Stage_3(x, h3)
    x, h4 = self.Stage_4(x, h4)

    return x, h1, h2, h3, h4

def Window_partition(x, local_patch_num):
  x = einops.rearrange(x, 'b c (yg yl) (xg xl) -> b (yg xg) (yl xl) c', yl = local_patch_num, xl = local_patch_num)

  return x
  
def Window_reverse(x, global_patch_num_x, local_patch_num):
  x = einops.rearrange(x, 'b (yl xl) (yg xg) c -> b c (yg yl) (xg xl)', yg = global_patch_num_x, yl = local_patch_num)

  return x

input_map = (1,256,256)
kernel_size = 2
local_patch_num = 4
channel_size = 128
num_head = 2
batch_size = 10

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

test = Modules(input_map, kernel_size, local_patch_num, channel_size, num_head)
test.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(test.parameters(), 0.01)

h1 = tuple(torch.empty((1,channel_size,1,1)).to(device, dtype = torch.float32) for _ in range(2))
h2 = tuple(torch.empty((1,channel_size,1,1)).to(device, dtype = torch.float32) for _ in range(2))
h3 = tuple(torch.empty((1,channel_size,1,1)).to(device, dtype = torch.float32) for _ in range(2))
h4 = tuple(torch.empty((1,channel_size,1,1)).to(device, dtype = torch.float32) for _ in range(2))

for i in tqdm(range(10)):
  x = torch.randn(batch_size, *input_map)
  y = torch.randn(batch_size, channel_size, 8, 8)
  x, y = x.to(device, dtype = torch.float32), y.to(device, dtype = torch.float32)

  yhat, h1, h2, h3, h4 = test(x, h1, h2, h3, h4)
  loss = criterion(yhat, y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  h1 = tuple(t.detach() for t in h1)
  h2 = tuple(t.detach() for t in h2)
  h3 = tuple(t.detach() for t in h3)
  h4 = tuple(t.detach() for t in h4)