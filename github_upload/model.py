import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class binary_classification_model(nn.Module):
    def __init__(self, max_seq_length, emb_length, proj_length, kernel_size, kernels, heads):
        super(binary_classification_model, self).__init__()
        self.kernels = kernels
        self.heads = heads
        self.projection = nn.Linear(in_features=emb_length, out_features=proj_length, bias=True)
        self.conv1 = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='same')
        self.attn = nn.Linear(in_features=int(proj_length / heads), out_features=max_seq_length, bias=True)
        self.classification = nn.Sequential(nn.Linear(in_features=proj_length, out_features=256, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=256, out_features=128, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=128, out_features=64, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=64, out_features=32, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=32, out_features=16, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=16, out_features=8, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=8, out_features=4, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=4, out_features=2, bias=True))
    def forward(self, x, seq_length, device):
        # x: [batch_size, max_seq_length, emb_length] -> [b, s, e],  b=1
        x = rearrange(x, 's e -> () s e') # [s, e] -> [b, s, e]
        representations_proj = self.projection(x) # [b, s, e] -> [b, s, p]
        representations_proj = rearrange(representations_proj, 'b s p -> b p s')
        l = int(seq_length.item())
        representations_proj[:, :, 0] = 0
        representations_proj[:, :, l+1:] = 0
        representations_conv = torch.zeros_like(representations_proj).to(device)
        representations_conv += F.relu(self.conv1(representations_proj))
        representations_conv += F.relu(self.conv2(representations_proj))
        representations_conv = rearrange(representations_conv, 'b (l h) s -> b h s l', h=self.heads)
        attn_score = self.attn(representations_conv) # [b, h, s, s]
        attn_score[:, :, :, 0] = -1e9
        attn_score[:, :, :, l+1:] = -1e9
        attn_score = F.softmax(attn_score, dim=-1)
        representations_attn = torch.matmul(attn_score, representations_conv) # [b, h, s, l]
        representations_attn = rearrange(representations_attn, 'b h s l -> b s (l h)') # [b, s, p]
        representations_attn[:, 0, :] = 0
        representations_attn[:, l+1:, :] = 0
        representations_attn = torch.sum(representations_attn, dim=1) # [b, p]
        representations_prob = F.softmax(self.classification(representations_attn), dim=1) # [b, 2]
        return representations_prob[:, 0], torch.max(representations_prob, dim=1).indices