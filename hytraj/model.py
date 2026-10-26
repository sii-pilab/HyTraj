from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from hytraj.controller import PIDController
from hytraj.transformer import PerceptionEncoder, WaypointStateDecoder, WaypointPerceptionDecoder


MEASUREMENT_DIM = 128


def value_head():
	return nn.Sequential(
		nn.Linear(256, 256),
		nn.ReLU(inplace=True),
		nn.Linear(256, 256),
		nn.Dropout(p=0.5),
		nn.ReLU(inplace=True),
		nn.Linear(256, 1),
	)


class HyTraj(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
		self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

		n_embd = config.n_embd

		self.measurements = nn.Sequential(
			nn.Linear(1 + 2 + 6, MEASUREMENT_DIM),
			nn.ReLU(inplace=True),
			nn.Linear(MEASUREMENT_DIM, MEASUREMENT_DIM),
			nn.ReLU(inplace=True),
		)

		self.encoder = PerceptionEncoder(
			n_embd=n_embd,
			n_head=config.n_head,
			block_exp=config.block_exp,
			n_layer=config.n_layer,
			n_cam=config.num_camera,
			vert_anchors=config.vert_anchors,
			horz_anchors=config.horz_anchors,
			seq_len=config.seq_len,
			embd_pdrop=config.embd_pdrop,
			attn_pdrop=config.attn_pdrop,
			resid_pdrop=config.resid_pdrop,
			pretrained=config.pretrained_backbone)

		self.speed_branch = nn.Sequential(
			nn.Linear(n_embd, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, 256),
			nn.Dropout(p=0.5),
			nn.ReLU(inplace=True),
			nn.Linear(256, 1),
		)

		self.join_traj = nn.Sequential(
			nn.Linear(n_embd + MEASUREMENT_DIM, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 256),
			nn.ReLU(inplace=True),
		)
		self.value_branch_traj = value_head()
		self.decoder_traj = nn.GRUCell(input_size=2 + 2, hidden_size=256)
		self.merge = nn.Sequential(
			nn.Linear(256 + n_embd, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 256),
		)
		self.output_traj = nn.Linear(256, 2)

		self.wp_emb = nn.Linear(2, n_embd)
		self.cross_wp_state = WaypointStateDecoder(
			n_embd=n_embd,
			n_head=config.n_head,
			block_exp=config.block_exp,
			n_layer=config.n_layer,
			pred_len=config.pred_len,
			measurement_dim=MEASUREMENT_DIM,
			embd_pdrop=config.embd_pdrop,
			attn_pdrop=config.attn_pdrop,
			resid_pdrop=config.resid_pdrop)
		self.wp_fuser_perception = WaypointPerceptionDecoder(
			n_embd=n_embd,
			n_head=config.n_head,
			block_exp=config.block_exp,
			n_layer=config.n_layer,
			attn_pdrop=config.attn_pdrop,
			resid_pdrop=config.resid_pdrop)
		self.network_wp = nn.Sequential(
			nn.Linear(n_embd, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 2),
		)
		self.join_ctrl = nn.Sequential(
			nn.Linear(n_embd + MEASUREMENT_DIM, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 256),
			nn.ReLU(inplace=True),
		)
		self.value_branch_ctrl = value_head()

	def forward(self, image, state, target_point, velocity, prev_waypoints):
		outputs = {}
		measurement_feature = self.measurements(state)

		perception_tokens = self.encoder(image, velocity)
		perception_feature = perception_tokens.mean(dim=1)

		wp_query = self.wp_emb(prev_waypoints)
		wp_state_feature = self.cross_wp_state(wp_query, measurement_feature)
		wp_feature = self.wp_fuser_perception(wp_state_feature, perception_tokens)
		outputs['wp_nar'] = self.network_wp(wp_feature)
		outputs['pred_speed'] = self.speed_branch(perception_feature)

		j_traj = self.join_traj(torch.cat([perception_feature, measurement_feature], 1))
		outputs['value_ar'] = self.value_branch_traj(j_traj)
		outputs['feature_ar'] = j_traj

		guide_feature = wp_state_feature[:, -1, :]
		z = j_traj
		x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).type_as(z)
		output_wp = []
		for i in range(self.config.pred_len):
			x_in = torch.cat([x, target_point], dim=1)
			z = self.decoder_traj(x_in, z)
			if i == self.config.pred_len - 1:
				h = self.merge(torch.cat([z, guide_feature], 1))
			else:
				h = z
			dx = self.output_traj(h)
			x = dx + x
			output_wp.append(x)
		outputs['wp_ar'] = torch.stack(output_wp, dim=1)

		j_ctrl = self.join_ctrl(torch.cat([perception_feature, measurement_feature], 1))
		outputs['value_nar'] = self.value_branch_ctrl(j_ctrl)
		outputs['feature_nar'] = j_ctrl
		return outputs

	def control_pid(self, waypoints, velocity, target):
		assert waypoints.size(0) == 1
		waypoints = waypoints[0].detach().cpu().numpy().copy()
		target = target.squeeze().detach().cpu().numpy().copy()

		waypoints[:, 1] *= -1
		target[1] *= -1

		num_pairs = len(waypoints) - 1
		best_norm = 1e5
		desired_speed = 0
		aim = waypoints[0]
		for i in range(num_pairs):
			desired_speed += np.linalg.norm(waypoints[i + 1] - waypoints[i]) * 2.0 / num_pairs
			norm = np.linalg.norm((waypoints[i + 1] + waypoints[i]) / 2.0)
			if abs(self.config.aim_dist - best_norm) > abs(self.config.aim_dist - norm):
				aim = waypoints[i]
				best_norm = norm

		aim_last = waypoints[-1] - waypoints[-2]
		angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
		angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
		angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

		use_target_to_aim = np.abs(angle_target) < np.abs(angle)
		use_target_to_aim = use_target_to_aim or (
			np.abs(angle_target - angle_last) > self.config.angle_thresh and target[1] < self.config.dist_thresh)
		angle_final = angle_target if use_target_to_aim else angle

		steer = np.clip(self.turn_controller.step(angle_final), -1.0, 1.0)
		speed = velocity[0].data.cpu().numpy()
		brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio
		delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
		throttle = np.clip(self.speed_controller.step(delta), 0.0, self.config.max_throttle)
		throttle = throttle if not brake else 0.0

		metadata = {
			'speed': float(speed.astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'wp_4': tuple(waypoints[3].astype(np.float64)),
			'wp_3': tuple(waypoints[2].astype(np.float64)),
			'wp_2': tuple(waypoints[1].astype(np.float64)),
			'wp_1': tuple(waypoints[0].astype(np.float64)),
			'aim': tuple(aim.astype(np.float64)),
			'target': tuple(target.astype(np.float64)),
			'desired_speed': float(desired_speed.astype(np.float64)),
			'angle': float(angle.astype(np.float64)),
			'angle_last': float(angle_last.astype(np.float64)),
			'angle_target': float(angle_target.astype(np.float64)),
			'angle_final': float(angle_final.astype(np.float64)),
			'delta': float(delta.astype(np.float64)),
		}
		return steer, throttle, brake, metadata


def torch_load(path, map_location='cpu'):
	try:
		return torch.load(path, map_location=map_location, weights_only=False)
	except TypeError:
		return torch.load(path, map_location=map_location)


def load_checkpoint(model, path, map_location='cpu'):
	ckpt = torch_load(path, map_location=map_location)
	state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
	model_keys = set(model.state_dict().keys())
	new_state_dict = OrderedDict()
	for key, value in state_dict.items():
		if key.startswith('model.'):
			key = key[len('model.'):]
		if key in model_keys:
			new_state_dict[key] = value
	missing = model_keys - set(new_state_dict.keys())
	if missing:
		raise RuntimeError('Missing keys when loading checkpoint: {}'.format(sorted(missing)))
	model.load_state_dict(new_state_dict, strict=True)
	return model
