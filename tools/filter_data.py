import argparse
import json
import os

import tqdm


def remove_files(root, index):
	items = {}
	sub_folders = list(os.listdir(root))
	for sub_folder in sub_folders:
		if len(list(os.listdir(os.path.join(root, sub_folder)))) == 0:
			break
		items[sub_folder] = '.' + list(os.listdir(os.path.join(root, sub_folder)))[0].split('.')[-1]
	for k, v in items.items():
		data_folder = os.path.join(root, k)
		total_len = len(os.listdir(data_folder))
		for i in range(index, total_len):
			file_name = os.path.join(data_folder, str(i).zfill(4) + v)
			os.remove(file_name)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--result_path', type=str, required=True)
	parser.add_argument('--data_path', type=str, required=True)
	args = parser.parse_args()

	routes_type = ['short']
	towns = ['town05', 'town06', 'town07', 'town10']
	result_pattern = 'routes_{}_{}.json'
	data_pattern = '{}_{}'

	for route_type in routes_type:
		print(route_type)
		for town in tqdm.tqdm(towns):
			result_file = result_pattern.format(town, route_type)
			data_folder = os.path.join(args.data_path, data_pattern.format(town, route_type))
			sub_folders = sorted(list(os.listdir(data_folder)))
			with open(os.path.join(args.result_path, result_file), 'r') as f:
				records = json.load(f)
			records = records['_checkpoint']['records']
			for index, record in enumerate(records):
				route_data_folder = os.path.join(data_folder, sub_folders[index])
				total_length = len(os.listdir(os.path.join(route_data_folder, 'measurements')))
				if record['scores']['score_composed'] >= 100:
					continue
				if len(record['infractions']['route_timeout']) > 0 or len(record['infractions']['vehicle_blocked']) > 0:
					stop_index = 0
					for i in range(total_length - 1, 0, -1):
						with open(os.path.join(route_data_folder, 'measurements', str(i).zfill(4)) + '.json', 'r') as mf:
							speed = json.load(mf)['speed']
							if speed > 0.1:
								stop_index = i
								break
					stop_index = min(total_length, stop_index + 20)
					remove_files(route_data_folder, stop_index)
				elif len(record['infractions']['red_light']) > 0 or \
					len(record['infractions']['collisions_pedestrian']) > 0 or \
					len(record['infractions']['collisions_vehicle']) > 0 or \
					len(record['infractions']['collisions_layout']) > 0:
					stop_index = max(0, total_length - 10)
					remove_files(route_data_folder, stop_index)
