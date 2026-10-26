import os


class GlobalConfig:
	seq_len = 1
	pred_len = 4

	root_dir_all = os.environ.get('HYTRAJ_DATA_ROOT', 'tcp_carla_data')

	train_towns = ['town01', 'town03', 'town04', 'town06']
	val_towns = ['town02', 'town05', 'town07', 'town10']
	train_data, val_data = [], []
	for town in train_towns:
		train_data.append(os.path.join(root_dir_all, town))
		train_data.append(os.path.join(root_dir_all, town + '_addition'))
	for town in val_towns:
		val_data.append(os.path.join(root_dir_all, town + '_val'))

	img_aug = True
	rl_ckpt = 'roach/log/ckpt_11833344.pth'

	pretrained_backbone = True
	num_camera = 1
	vert_anchors = 8
	horz_anchors = 8

	n_layer = 2
	n_embd = 512
	block_exp = 4
	n_head = 4
	embd_pdrop = 0.1
	resid_pdrop = 0.1
	attn_pdrop = 0.1

	features_weight = 0.05
	ar_wp_weight = 0.5
	nar_wp_weight = 0.5
	ar_value_weight = 0.01
	nar_value_weight = 0.01
	speed_weight = 0.05

	turn_KP = 0.75
	turn_KI = 0.75
	turn_KD = 0.3
	turn_n = 40

	speed_KP = 5.0
	speed_KI = 0.5
	speed_KD = 1.0
	speed_n = 40

	max_throttle = 0.75
	brake_speed = 0.4
	brake_ratio = 1.1
	clip_delta = 0.25

	aim_dist = 4.0
	angle_thresh = 0.3
	dist_thresh = 10

	fusion_mode = 'hybrid'
	fusion_alpha = 0.3
	turn_window = 20
	turn_steer_thresh = 0.10
	turn_ratio_thresh = 0.5

	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)
