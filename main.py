# Converted from ipynb file
import os
import random
import bisect
import tarfile
from glob import glob
from time import sleep

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn # nn.BCELoss
import torch.optim as optimizer
import torch.nn.functional as F 
import torch.utils.data as data # data.DataLoader, data.Dataset
import torchvision.transforms as transforms # ToTensor
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)
plt.style.use('ggplot')



model_name = 'UMCD_AIQU_v1.pth'
state_dict_path = 'data/models/' + model_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Model name: {model_name}, Using device {device}')

tarfile.open('data/umcd-normal.tgz', 'r:gz').extractall()
video_folder = 'data/Normal/'



class UMCD_video_dset(data.Dataset):
  def __init__(self, video_folder, tfms, target_chw=(3,256,256), n_input_frames=6, specific_videos=None):
    self.video_paths = glob(os.path.join(video_folder, '*.mp4')) if not specific_videos else specific_videos
    self.tfms, self.target_chw = tfms, target_chw

    self.n_input_frames = n_input_frames
    self.n_pred_frames = 1
    self.n = self.n_input_frames + self.n_pred_frames

    self.setup()


  def setup(self):
    frame_intervals = []
    for video_path in self.video_paths:
      print(video_path)
      cap = cv2.VideoCapture(video_path)
      fs = cap.get(cv2.CAP_PROP_FRAME_COUNT)
      fs = int((fs // self.n) * self.n)

      if len(frame_intervals) == 0: frame_intervals = [fs // self.n]
      else: frame_intervals.append(frame_intervals[-1] + fs // self.n)
      cap.release()

    self.frame_intervals = frame_intervals
    

  def __getitem__(self, batch_idx):
    """
    Returns a tuple of (input frames, target frame)
    """
    frame_idxs = list(map(lambda x: x + batch_idx * self.n, range(self.n)))
    video_idx = bisect.bisect_right(self.frame_intervals, batch_idx)
    all_frames = torch.empty(self.n, self.target_chw[0], self.target_chw[1], self.target_chw[2])
    
    cap = cv2.VideoCapture(self.video_paths[video_idx])
    for i, frame_idx in enumerate(frame_idxs):
      tup = cap.read(frame_idx)
      assert tup[0] is not None, 'Failed to extract videoframe from videocapture'
      tfmd_frame = self.tfms(tup[1])
      all_frames[i] = tfmd_frame
    cap.release()

    input_frames, target_frames = all_frames[:self.n_input_frames], all_frames[self.n_input_frames:]
    return (input_frames, target_frames.squeeze())


  def __len__(self): 
    return self.frame_intervals[-1]


  def show(self, batch_idx=None):
    batch_idx = batch_idx if batch_idx else random.randint(0, self.frame_intervals[-1])
    tup = self.__getitem__(batch_idx)
    input_frames, target_frame = tup[0].numpy(), tup[1].numpy()

    f, axs = plt.subplots(1, self.n, figsize=(self.n*3, 3))

    for i, img in enumerate(input_frames): 
      axs[i].imshow(cv2.cvtColor(img.transpose(1,2,0), cv2.COLOR_BGR2RGB))
      axs[i].set_title(f'Input frame {i+1}')
      axs[i].axis('off')

    img = target_frame
    i = self.n_input_frames
    axs[i].imshow(cv2.cvtColor(img.transpose(1,2,0), cv2.COLOR_BGR2RGB))
    axs[i].set_title(f'Target frame')
    axs[i].axis('off')

    plt.tight_layout()
    plt.show()

class Linear(nn.Module):
  def __init__(self, input_dim=(768,), output_dim=(6, 768//6)):
    super(Linear, self).__init__()

    self.weight = nn.Parameter(torch.randn(*input_dim, *output_dim))
    self.bias = nn.Parameter(torch.randn(*output_dim))

  def forward(self, x, dims):
    return torch.tensordot(x, self.weight, dims=dims) + self.bias

class Multihead_Self_Attention(nn.Module):
  def __init__(self, input_dim=768, n_attention_heads=6):
    super(Multihead_Self_Attention, self).__init__()

    self.input_dim, self.n_attention_heads = input_dim, n_attention_heads

    assert input_dim % n_attention_heads == 0, f'Input dimension is not divisible by number of attention heads'
    
    self.head_dim = input_dim // n_attention_heads

    self.query = Linear(input_dim=(input_dim,), output_dim=(n_attention_heads, self.head_dim))
    self.key = Linear(input_dim=(input_dim,), output_dim=(n_attention_heads, self.head_dim))
    self.value = Linear(input_dim=(input_dim,), output_dim=(n_attention_heads, self.head_dim))
    self.out = Linear(input_dim=(n_attention_heads, self.head_dim), output_dim=(input_dim,))
    
  def forward(self, x):
    # (N, n_embeddings, embedding_dim) --> (N, n_embeddings, n_attention_heads, head_dim) --> (N, n_embeddings, embedding_dim)

    query = self.query(x, dims=([2], [0])).permute(0, 2, 1, 3)
    key = self.key(x, dims=([2], [0])).permute(0, 2, 1, 3)
    value = self.value(x, dims=([2], [0])).permute(0, 2, 1, 3)

    attn_weights = torch.matmul(query, key.transpose(-2,-1)) / self.head_dim ** 0.5
    attn_weights = F.softmax(attn_weights, dim=-1)

    out = torch.matmul(attn_weights, value).permute(0, 2, 1, 3)

    return self.out(out, dims=([2,3],[0,1]))

class VisionTransformer_block(nn.Module):
  def __init__(self, input_dim=768, mlp_dim=4096, n_attention_heads=6):
    super(VisionTransformer_block, self).__init__()

    self.norm1 = nn.LayerNorm(input_dim)
    self.attention = Multihead_Self_Attention(input_dim, n_attention_heads)
    self.dropout = nn.Dropout(0.1)
    self.norm2 = nn.LayerNorm(input_dim)
    self.mlp = nn.Sequential(
        nn.Linear(input_dim, mlp_dim),
        nn.GELU(),
        nn.Linear(mlp_dim, input_dim)
    )

  def forward(self, x):
    residual = x
    out = self.norm1(x)
    out = self.attention(x)
    out = self.dropout(out)
    out = out + residual

    residual = out
    out = self.norm2(out)
    out = self.mlp(out)
    out = out + residual

    return out

class SpatioTemporal_Encoder(nn.Module):
  def __init__(self, frame_size=(256, 256), patch_size=(16, 16), emb_dim=768, mlp_dim=1024, n_heads=2, n_layers=2, n_frames=6):
    super(SpatioTemporal_Encoder, self).__init__()

    H, W = frame_size
    h, w = patch_size

    self.n_patches = (H // h) * (W // w) * n_frames

    self.patchEmbed = nn.Conv2d(3, emb_dim, kernel_size=(h, w), stride=(h, w))

    self.clsEmbed = nn.Parameter(torch.zeros(1, 1, emb_dim))

    self.posEmbeds = nn.Parameter(torch.randn(1, self.n_patches + 1, emb_dim))

    self.transformer_layers = nn.ModuleList([VisionTransformer_block(emb_dim, mlp_dim, n_heads) for _ in range(n_layers)])

    
  def forward(self, x):
    """
    x = videos with shape (N, n_frames, n_channels, height, width) 
    """
    for i in range(x.shape[1]):
      e = self.patchEmbed(x[:, i])
      e = e.permute(0, 2, 3, 1)
      n, h, w, c = e.shape
      e = e.reshape(n, h*w, c)
      patchEmbeds = e if i == 0 else torch.cat((e, patchEmbeds), 1)

    n, m, c = patchEmbeds.shape

    clsEmbed = self.clsEmbed.repeat(n, 1, 1)

    embs = torch.cat([clsEmbed, patchEmbeds], dim=1)

    feat = embs + self.posEmbeds

    for transformer_layer in self.transformer_layers:
      feat = transformer_layer(feat)
 
    return feat

class SpatioTemporal_Decoder(nn.Module):
  def __init__(self, frame_size=(256, 256), patch_size=(16, 16), emb_dim=768, n_frames=6):
    super(SpatioTemporal_Decoder, self).__init__()

    H, W = frame_size
    h, w = patch_size

    self.n_patches = (H // h) * (W // w) * n_frames
    self.rest = emb_dim // (h*w)

    assert self.rest == emb_dim / (h*w), 'Not. SO. FAST. !!! faulty embedding dimension!'

    def conv(cIn, cOut):
      return nn.Sequential(
        nn.Conv2d(cIn, cOut, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(cOut),
        nn.ReLU(inplace=False),
        nn.Conv2d(cOut, cOut, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(cOut),
        nn.ReLU(inplace=False),
      )

    def transConv(cIn, cOut):
      return nn.Sequential(
        nn.ConvTranspose2d(cIn, cOut, kernel_size=2, stride=2, padding=0, output_padding=0),
        nn.BatchNorm2d(cOut),
        nn.ReLU(inplace=False),
      )

    def output(cIn, cMiddle, cOut):
      return nn.Sequential(
        nn.Conv2d(cIn, cMiddle, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(cMiddle),
        nn.ReLU(inplace=False),
        nn.Conv2d(cMiddle, cOut, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(cOut),
        nn.Sigmoid(),
      )

    self.decoder = nn.Sequential(
      conv((self.n_patches + 1) * self.rest, 256),
      transConv(256, 128),
      transConv(128, 64),
      transConv(64, 32),
      transConv(32, 16),
      output(16,16,3)      
    )
    
  def forward(self, x):
    """
    x.shape = torch.Size([7, 1537, 768])
    """
    out = x.view(x.size(0), x.size(1) * self.rest, 16, 16)
    out = self.decoder(out)
    return out

class SpatioTemporal_AE(nn.Module):
  def __init__(self, encoder, decoder):
    super(SpatioTemporal_AE, self).__init__()
    
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, videos):
    encoded_input = self.encoder(videos)
    decoded_output = self.decoder(encoded_input)
    return decoded_output


target_chw=(3, 256, 256)
tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((target_chw[1], target_chw[2])),
    transforms.ToTensor(),
    ])

dset = UMCD_video_dset(video_folder, tfms=tfms, target_chw=target_chw, n_input_frames=6)
train, test = data.random_split(dset, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
train_dl = data.DataLoader(train, batch_size = 32, shuffle = False)
test_dl = data.DataLoader(test, batch_size = 32, shuffle = False)



frame_size = (256, 256)
patch_size = (16, 16)

emb_dim = 512
mlp_dim = 2048

n_heads = 8
n_layers = 2
n_frames = 3

encoder = SpatioTemporal_Encoder(frame_size=frame_size, patch_size=patch_size, emb_dim=emb_dim, mlp_dim=mlp_dim, n_heads=n_heads, n_layers=n_layers, n_frames=n_frames).to(device)
decoder = SpatioTemporal_Decoder(frame_size=frame_size, patch_size=patch_size, emb_dim=emb_dim, n_frames=n_frames).to(device)
model = SpatioTemporal_AE(encoder, decoder).to(device)

trainable_params = filter(lambda p: p.requires_grad, model.parameters())
trainable_params = sum([np.prod(p.size()) for p in trainable_params])
print(f'Trainable params: {trainable_params:_}')

try:
  model_weights = 'UMCD_AIQU_v1.pth'
  weights_path = 'data/' + model_weights
  model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)), strict=False)
  print('Pretrained weights loaded sucessfully\n')
except:
  print('Could not load pretrained weights\n')

sample_input = torch.rand(7, n_frames, 3, frame_size[0], frame_size[0]).to(device)
sample_output = model(sample_input)

print('-'*20,model_name,'-'*20)
print(f'Model parameters:\nframe_size:{frame_size}, patch_size:{patch_size}\nemb_dim:{emb_dim}, mlp_dim:{mlp_dim}, n_heads:{n_heads}, n_layers:{n_layers}\n')
print(f'Sample input shape: {sample_input.shape}\nSample output shape: {sample_output.shape}')
print(f'Sample output values ranging from {torch.min(sample_output).item():.5f} to {torch.max(sample_output).item():.5f}\n')

torch.cuda.empty_cache()

lr = 5e-3
epochs = 10
optim = optimizer.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.BCELoss()



def show_preds(n=1):
  for _ in range(n):
    data = random.choice(test_dl.dataset) 

    input = data[0].to(device)
    target = data[1].to(device)
    pred = model(input.unsqueeze(0).to(device))

    n = len(input) + 2
    f, axs = plt.subplots(1, n, figsize=(n*3, 3))

    for i, img in enumerate(input): 
      axs[i].imshow(cv2.cvtColor(img.cpu().numpy().transpose(1,2,0), cv2.COLOR_BGR2RGB))
      axs[i].set_title(f'Input frame {i+1}')
      axs[i].axis('off')

    img = pred[0]
    axs[n-2].imshow(cv2.cvtColor(img.squeeze().detach().cpu().numpy().transpose(1,2,0), cv2.COLOR_BGR2RGB))
    axs[n-2].set_title(f'Predicted frame {}')
    axs[n-2].axis('off')

    img = target
    axs[n-1].imshow(cv2.cvtColor(img.squeeze().detach().cpu().numpy().transpose(1,2,0), cv2.COLOR_BGR2RGB))
    axs[n-1].set_title(f'Observed frame {7}')
    axs[n-1].axis('off')
    

    plt.tight_layout()
    plt.show()

def train_epochs(epochs=5):
  print(f'{"-"*20}Training model {model_name} for {epochs} epochs with {trainable_params:_} trainable params and learning rate of {lr}{"-"*20}\n')
  min_test_loss = np.inf

  for epoch in range(epochs):
    train_epoch_loss = 0.
    model.train()

    with tqdm(train_dl, unit="batch") as tepoch:
      for batch in tepoch:
        tepoch.set_description(f"Training Epoch {epoch + 1} / {epochs}")

        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = model(x)

        train_loss = criterion(y_hat, y)
        train_epoch_loss += train_loss.detach().cpu().item() / len(train_dl)

        optim.zero_grad()
        train_loss.backward()
        optim.step()

        tepoch.set_postfix(loss=train_loss.item())
        sleep(0.1)


    with torch.no_grad():
      test_epoch_loss = 0
      model.eval()

      with tqdm(test_dl, unit="batch") as tepoch:
        for batch in tepoch:
          tepoch.set_description(f"Testing Epoch {epoch + 1} / {epochs}")

          x, y = batch
          x, y = x.to(device), y.to(device)
          y_hat = model(x)

          test_loss = criterion(y_hat, y)
          test_epoch_loss += test_loss.detach().cpu().item() / len(test_dl)

          tepoch.set_postfix(loss=test_loss.item())
          sleep(0.1)

      print(f'[Epoch {epoch + 1}]\tTrain loss: {train_epoch_loss:.4f}\tTest loss {test_epoch_loss:.4f}\nSample prediction:\n')
      show_preds()
      print('')
      if test_epoch_loss < min_test_loss:
        print(f'Test loss decreased ({min_test_loss:.4f}--->{test_epoch_loss:.4f})\tSaving the model...\n')
        min_test_loss = test_epoch_loss
        torch.save(model.state_dict(), state_dict_path)
      else: print('')



train_epochs(epochs)

