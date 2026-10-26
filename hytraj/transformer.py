import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


def normalize_imagenet(x):
	x = x.clone()
	x[:, 0] = (x[:, 0] - 0.485) / 0.229
	x[:, 1] = (x[:, 1] - 0.456) / 0.224
	x[:, 2] = (x[:, 2] - 0.406) / 0.225
	return x


def init_transformer_weights(module):
	if isinstance(module, nn.Linear):
		module.weight.data.normal_(mean=0.0, std=0.02)
		if module.bias is not None:
			module.bias.data.zero_()
	elif isinstance(module, nn.LayerNorm):
		module.bias.data.zero_()
		module.weight.data.fill_(1.0)


def sinusoidal_embedding(length, dim):
	position = torch.arange(0, length).unsqueeze(1)
	div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
	table = torch.zeros(length, dim)
	table[:, 0::2] = torch.sin(position * div_term)
	table[:, 1::2] = torch.cos(position * div_term)
	return table.unsqueeze(0)


class ImageCNN(nn.Module):
	def __init__(self, vert_anchors, horz_anchors, normalize=True, pretrained=True):
		super().__init__()
		self.normalize = normalize
		self.avgpool = nn.AdaptiveAvgPool2d((vert_anchors, horz_anchors))
		self.features = models.resnet34(pretrained=pretrained)

	def forward(self, x):
		if self.normalize:
			x = normalize_imagenet(x)
		x = self.features.conv1(x)
		x = self.features.bn1(x)
		x = self.features.relu(x)
		x = self.features.maxpool(x)
		x = self.features.layer1(x)
		x = self.features.layer2(x)
		x = self.features.layer3(x)
		x = self.features.layer4(x)
		x = self.avgpool(x)
		return x


class MultiHeadAttention(nn.Module):
	def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
		super().__init__()
		assert n_embd % n_head == 0
		self.key = nn.Linear(n_embd, n_embd)
		self.query = nn.Linear(n_embd, n_embd)
		self.value = nn.Linear(n_embd, n_embd)
		self.attn_drop = nn.Dropout(attn_pdrop)
		self.resid_drop = nn.Dropout(resid_pdrop)
		self.proj = nn.Linear(n_embd, n_embd)
		self.n_head = n_head

	def _split_heads(self, x):
		B, T, C = x.size()
		return x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

	def forward(self, x, context):
		B, T, C = x.size()
		q = self._split_heads(self.key(x))
		k = self._split_heads(self.query(context))
		v = self._split_heads(self.value(context))
		att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
		att = F.softmax(att, dim=-1)
		att = self.attn_drop(att)
		y = att @ v
		y = y.transpose(1, 2).contiguous().view(B, T, C)
		y = self.resid_drop(self.proj(y))
		return y


class SelfAttention(nn.Module):
	def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
		super().__init__()
		assert n_embd % n_head == 0
		self.key = nn.Linear(n_embd, n_embd)
		self.query = nn.Linear(n_embd, n_embd)
		self.value = nn.Linear(n_embd, n_embd)
		self.attn_drop = nn.Dropout(attn_pdrop)
		self.resid_drop = nn.Dropout(resid_pdrop)
		self.proj = nn.Linear(n_embd, n_embd)
		self.n_head = n_head

	def forward(self, x):
		B, T, C = x.size()
		k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
		q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
		v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
		att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
		att = F.softmax(att, dim=-1)
		att = self.attn_drop(att)
		y = att @ v
		y = y.transpose(1, 2).contiguous().view(B, T, C)
		y = self.resid_drop(self.proj(y))
		return y


class SelfAttentionBlock(nn.Module):
	def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
		super().__init__()
		self.ln1 = nn.LayerNorm(n_embd)
		self.ln2 = nn.LayerNorm(n_embd)
		self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
		self.mlp = nn.Sequential(
			nn.Linear(n_embd, block_exp * n_embd),
			nn.ReLU(True),
			nn.Linear(block_exp * n_embd, n_embd),
			nn.Dropout(resid_pdrop),
		)

	def forward(self, x):
		x = x + self.attn(self.ln1(x))
		x = x + self.mlp(self.ln2(x))
		return x


class CrossAttentionBlock(nn.Module):
	def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
		super().__init__()
		self.ln1 = nn.LayerNorm(n_embd)
		self.ln2 = nn.LayerNorm(n_embd)
		self.ln3 = nn.LayerNorm(n_embd)
		self.attn = MultiHeadAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
		self.mlp = nn.Sequential(
			nn.Linear(n_embd, block_exp * n_embd),
			nn.ReLU(True),
			nn.Linear(block_exp * n_embd, n_embd),
			nn.Dropout(resid_pdrop),
		)

	def forward(self, x, context):
		x = x + self.attn(self.ln1(x), self.ln2(context))
		x = x + self.mlp(self.ln3(x))
		return x


class PerceptionEncoder(nn.Module):
	def __init__(self, n_embd, n_head, block_exp, n_layer, n_cam, vert_anchors, horz_anchors, seq_len,
				 embd_pdrop, attn_pdrop, resid_pdrop, pretrained=True):
		super().__init__()
		self.n_embd = n_embd
		self.img_emb = ImageCNN(vert_anchors, horz_anchors, normalize=True, pretrained=pretrained)
		self.pos_emb = nn.Parameter(torch.zeros(1, n_cam * seq_len * vert_anchors * horz_anchors, n_embd))
		self.vel_emb = nn.Linear(1, n_embd)
		self.drop = nn.Dropout(embd_pdrop)
		self.blocks = nn.Sequential(*[SelfAttentionBlock(n_embd, n_head, block_exp, attn_pdrop, resid_pdrop)
									  for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embd)
		self.apply(init_transformer_weights)

	def forward(self, image, velocity):
		batch_size = velocity.size(0)
		image_embeddings = self.img_emb(image)
		image_embeddings = image_embeddings.view(batch_size, self.n_embd, -1).transpose(1, 2)
		velocity_embeddings = self.vel_emb(velocity.unsqueeze(1))
		x = self.drop(self.pos_emb + image_embeddings + velocity_embeddings.unsqueeze(1))
		x = self.blocks(x)
		x = self.ln_f(x)
		return x


class WaypointStateDecoder(nn.Module):
	def __init__(self, n_embd, n_head, block_exp, n_layer, pred_len, measurement_dim,
				 embd_pdrop, attn_pdrop, resid_pdrop):
		super().__init__()
		self.drop = nn.Dropout(embd_pdrop)
		self.blocks = nn.Sequential(*[CrossAttentionBlock(n_embd, n_head, block_exp, attn_pdrop, resid_pdrop)
									  for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embd)
		self.apply(init_transformer_weights)
		self.pos_wp_emb = nn.Parameter(torch.zeros(1, pred_len, n_embd))
		self.mea_emb = nn.Sequential(
			nn.Linear(measurement_dim, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, n_embd),
			nn.ReLU(inplace=True),
		)
		self.register_buffer('sin_cos_wp_emb', sinusoidal_embedding(pred_len, n_embd))

	def forward(self, wp_feature, measurement):
		x = self.drop(wp_feature + self.sin_cos_wp_emb)
		context = self.mea_emb(measurement)
		context = self.drop(context.unsqueeze(1) + self.pos_wp_emb + x)
		for block in self.blocks:
			x = block(x, context)
		x = self.ln_f(x)
		return x


class WaypointPerceptionDecoder(nn.Module):
	def __init__(self, n_embd, n_head, block_exp, n_layer, attn_pdrop, resid_pdrop):
		super().__init__()
		self.blocks = nn.Sequential(*[CrossAttentionBlock(n_embd, n_head, block_exp, attn_pdrop, resid_pdrop)
									  for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embd)
		self.apply(init_transformer_weights)

	def forward(self, wp_feature, perception):
		x = wp_feature
		for block in self.blocks:
			x = block(x, perception)
		x = self.ln_f(x)
		return x
