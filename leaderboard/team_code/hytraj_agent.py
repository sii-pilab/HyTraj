import os
import json
import datetime
import pathlib
import time
import math
from collections import deque

import cv2
import carla
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from leaderboard.autoagents import autonomous_agent
from hytraj.config import GlobalConfig
from hytraj.model import HyTraj, load_checkpoint
from team_code.planner import RoutePlanner


SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
	return 'HyTrajAgent'


class HyTrajAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file):
		self.track = autonomous_agent.Track.SENSORS
		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		self.config = GlobalConfig(pretrained_backbone=False)
		self.config.fusion_mode = os.environ.get('FUSION_MODE', self.config.fusion_mode)
		self.config.fusion_alpha = float(os.environ.get('FUSION_ALPHA', self.config.fusion_alpha))
		assert self.config.fusion_mode in ('hybrid', 'ar', 'nar')

		self.net = HyTraj(self.config)
		load_checkpoint(self.net, path_to_conf_file)
		self.net.cuda()
		self.net.eval()

		self.prev_wp = torch.zeros((1, self.config.pred_len, 2), dtype=torch.float32).to('cuda')
		self.turning = False
		self.last_steers = deque()

		self._im_transform = T.Compose(
			[T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

		self.save_path = None
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
			print(string)
			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
			self.save_path.mkdir(parents=True, exist_ok=False)
			(self.save_path / 'rgb').mkdir()
			(self.save_path / 'meta').mkdir()
			(self.save_path / 'bev').mkdir()

	def _init(self):
		self._route_planner = RoutePlanner(4.0, 50.0)
		self._route_planner.set_route(self._global_plan, True)
		self.initialized = True

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale
		return gps

	def sensors(self):
		return [
			{
				'type': 'sensor.camera.rgb',
				'x': -1.5, 'y': 0.0, 'z': 2.0,
				'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
				'width': 900, 'height': 256, 'fov': 100,
				'id': 'rgb'
			},
			{
				'type': 'sensor.camera.rgb',
				'x': 0.0, 'y': 0.0, 'z': 50.0,
				'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
				'width': 512, 'height': 512, 'fov': 5 * 10.0,
				'id': 'bev'
			},
			{
				'type': 'sensor.other.imu',
				'x': 0.0, 'y': 0.0, 'z': 0.0,
				'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
				'sensor_tick': 0.05,
				'id': 'imu'
			},
			{
				'type': 'sensor.other.gnss',
				'x': 0.0, 'y': 0.0, 'z': 0.0,
				'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
				'sensor_tick': 0.01,
				'id': 'gps'
			},
			{
				'type': 'sensor.speedometer',
				'reading_frequency': 20,
				'id': 'speed'
			}
		]

	def tick(self, input_data):
		self.step += 1
		rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]
		if math.isnan(compass):
			compass = 0.0

		result = {'rgb': rgb, 'gps': gps, 'speed': speed, 'compass': compass, 'bev': bev}
		pos = self._get_position(result)
		result['gps'] = pos
		next_wp, next_cmd = self._route_planner.run_step(pos)
		result['next_command'] = next_cmd.value

		theta = compass + np.pi / 2
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)]
		])
		local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result['target_point'] = tuple(local_command_point)
		return result

	def _fusion_weights(self):
		if self.config.fusion_mode == 'ar':
			return 0.0, 1.0
		if self.config.fusion_mode == 'nar':
			return 1.0, 0.0
		alpha = self.config.fusion_alpha
		if self.turning:
			return 1 - alpha, alpha
		return alpha, 1 - alpha

	def _update_turning(self, steer):
		if len(self.last_steers) >= self.config.turn_window:
			self.last_steers.popleft()
		self.last_steers.append(abs(float(steer)))
		num = sum(1 for s in self.last_steers if s > self.config.turn_steer_thresh)
		self.turning = num > self.config.turn_ratio_thresh * self.config.turn_window

	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()
		tick_data = self.tick(input_data)

		if self.step < self.config.seq_len:
			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			return control

		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
		command = tick_data['next_command']
		if command < 0:
			command = 4
		command -= 1
		assert command in [0, 1, 2, 3, 4, 5]
		cmd_one_hot = [0] * 6
		cmd_one_hot[command] = 1
		cmd_one_hot = torch.tensor(cmd_one_hot).view(1, 6).to('cuda', dtype=torch.float32)
		speed = torch.FloatTensor([float(tick_data['speed'])]).view(1, 1).to('cuda', dtype=torch.float32) / 12
		rgb = self._im_transform(tick_data['rgb']).unsqueeze(0).to('cuda', dtype=torch.float32)
		target_point = torch.FloatTensor([tick_data['target_point']]).to('cuda', dtype=torch.float32)
		state = torch.cat([speed, target_point, cmd_one_hot], 1)

		pred = self.net(rgb, state, target_point, gt_velocity, self.prev_wp)
		self.prev_wp = pred['wp_nar']

		steer_ar, throttle_ar, brake_ar, metadata_ar = self.net.control_pid(pred['wp_ar'], gt_velocity, target_point)
		steer_nar, throttle_nar, brake_nar, metadata_nar = self.net.control_pid(pred['wp_nar'], gt_velocity, target_point)

		if brake_ar < 0.05:
			brake_ar = 0.0
		if throttle_ar > brake_ar:
			brake_ar = 0.0
		if brake_nar < 0.05:
			brake_nar = 0.0
		if throttle_nar > brake_nar:
			brake_nar = 0.0

		w_nar, w_ar = self._fusion_weights()
		control = carla.VehicleControl()
		control.steer = np.clip(w_nar * steer_nar + w_ar * steer_ar, -1, 1)
		control.throttle = np.clip(w_nar * throttle_nar + w_ar * throttle_ar, 0, 0.75)
		control.brake = np.clip(w_nar * brake_nar + w_ar * brake_ar, 0, 1)
		if control.brake > 0.5:
			control.throttle = float(0)

		self.pid_metadata = metadata_ar
		self.pid_metadata['nar'] = metadata_nar
		self.pid_metadata['turning'] = int(self.turning)
		self.pid_metadata['w_nar'] = float(w_nar)
		self._update_turning(control.steer)

		if SAVE_PATH is not None and self.step % 10 == 0:
			self.save(tick_data)
		return control

	def save(self, tick_data):
		frame = self.step // 10
		Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))
		Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))
		with open(self.save_path / 'meta' / ('%04d.json' % frame), 'w') as outfile:
			json.dump(self.pid_metadata, outfile, indent=4)

	def destroy(self):
		del self.net
		torch.cuda.empty_cache()
