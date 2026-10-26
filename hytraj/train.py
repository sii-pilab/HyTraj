import argparse
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from hytraj.config import GlobalConfig
from hytraj.data import CarlaDataset
from hytraj.model import HyTraj, torch_load


class HyTrajPlanner(pl.LightningModule):
	def __init__(self, config, lr, encoder_ckpt=None):
		super().__init__()
		self.lr = lr
		self.config = config
		self.model = HyTraj(config)
		self._load_roach_value_heads()
		if encoder_ckpt:
			self._freeze_encoder(encoder_ckpt)
		self.train_prev_wp = None
		self.val_prev_wp = None

	def _load_roach_value_heads(self):
		rl_state_dict = torch_load(self.config.rl_ckpt, map_location='cpu')['policy_state_dict']
		self._load_state_dict(self.model.value_branch_traj, rl_state_dict, 'value_head')
		self._load_state_dict(self.model.value_branch_ctrl, rl_state_dict, 'value_head')

	def _load_state_dict(self, il_net, rl_state_dict, key_word):
		rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
		il_keys = il_net.state_dict().keys()
		assert len(rl_keys) == len(il_keys)
		new_state_dict = OrderedDict()
		for k_il, k_rl in zip(il_keys, rl_keys):
			new_state_dict[k_il] = rl_state_dict[k_rl]
		il_net.load_state_dict(new_state_dict)

	def _freeze_encoder(self, encoder_ckpt):
		self.model.encoder.load_state_dict(torch_load(encoder_ckpt, map_location='cpu'), strict=True)
		for param in self.model.encoder.parameters():
			param.requires_grad = False
		for module in self.model.encoder.modules():
			if isinstance(module, nn.modules.batchnorm._BatchNorm):
				module.momentum = 0.0

	def forward(self, batch):
		pass

	def _shared_step(self, batch, prev_wp):
		front_img = batch['front_img']
		ego_speed = batch['speed'].to(dtype=torch.float32).view(-1)
		speed = ego_speed.view(-1, 1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']
		state = torch.cat([speed, target_point, command], 1)
		value = batch['value'].view(-1, 1)
		feature = batch['feature']
		gt_waypoints = batch['waypoints']

		if prev_wp is None or prev_wp.size(0) != front_img.size(0):
			prev_wp = torch.zeros((front_img.size(0), self.config.pred_len, 2), dtype=torch.float32, device=front_img.device)

		pred = self.model(front_img, state, target_point, ego_speed, prev_wp)

		losses = OrderedDict()
		losses['speed_loss'] = F.mse_loss(pred['pred_speed'], speed) * self.config.speed_weight
		losses['value_loss_ar'] = F.mse_loss(pred['value_ar'], value) * self.config.ar_value_weight
		losses['value_loss_nar'] = F.mse_loss(pred['value_nar'], value) * self.config.nar_value_weight
		losses['feature_loss'] = (F.mse_loss(pred['feature_nar'], feature) + F.mse_loss(pred['feature_ar'], feature)) * self.config.features_weight
		losses['wp_loss_ar'] = F.l1_loss(pred['wp_ar'], gt_waypoints) * self.config.ar_wp_weight
		losses['wp_loss_nar'] = F.l1_loss(pred['wp_nar'], gt_waypoints) * self.config.nar_wp_weight
		losses['loss'] = sum(losses.values())
		return losses, pred['wp_nar'].detach()

	def training_step(self, batch, batch_idx):
		losses, self.train_prev_wp = self._shared_step(batch, self.train_prev_wp)
		for name, value in losses.items():
			self.log('train_' + name, value.item(), sync_dist=True)
		return losses['loss']

	def validation_step(self, batch, batch_idx):
		losses, self.val_prev_wp = self._shared_step(batch, self.val_prev_wp)
		for name, value in losses.items():
			self.log('val_' + name, value.item(), sync_dist=True)

	def configure_optimizers(self):
		params = [p for p in self.parameters() if p.requires_grad]
		optimizer = optim.Adam(params, lr=self.lr, weight_decay=1e-7)
		lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 50, 0.5)
		return [optimizer], [lr_scheduler]


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=str, default='hytraj')
	parser.add_argument('--epochs', type=int, default=80)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--val_every', type=int, default=3)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--logdir', type=str, default='log')
	parser.add_argument('--gpus', type=int, default=4)
	parser.add_argument('--encoder_ckpt', type=str, default=None)
	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)
	os.makedirs(args.logdir, exist_ok=True)

	config = GlobalConfig()
	train_set = CarlaDataset(root=config.root_dir_all, data_folders=config.train_data, img_aug=config.img_aug)
	val_set = CarlaDataset(root=config.root_dir_all, data_folders=config.val_data)
	print(len(train_set), len(val_set))

	dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
	dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)

	planner = HyTrajPlanner(config, args.lr, encoder_ckpt=args.encoder_ckpt)
	checkpoint_callback = ModelCheckpoint(
		save_weights_only=False, mode='min', monitor='val_loss', save_top_k=2, save_last=True,
		dirpath=args.logdir, filename='best_{epoch:02d}-{val_loss:.3f}')
	checkpoint_callback.CHECKPOINT_NAME_LAST = '{epoch}-last'
	trainer = pl.Trainer.from_argparse_args(
		args,
		default_root_dir=args.logdir,
		gpus=args.gpus,
		accelerator='ddp',
		sync_batchnorm=True,
		plugins=DDPPlugin(find_unused_parameters=True),
		profiler='simple',
		benchmark=True,
		log_every_n_steps=1,
		flush_logs_every_n_steps=5,
		callbacks=[checkpoint_callback],
		check_val_every_n_epoch=args.val_every,
		max_epochs=args.epochs)

	trainer.fit(planner, dataloader_train, dataloader_val)
	if trainer.is_global_zero and not args.encoder_ckpt:
		torch.save(planner.model.encoder.state_dict(), os.path.join(args.logdir, 'encoder.pth'))
