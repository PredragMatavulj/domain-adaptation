import os
import pickle
import numpy as np
import pandas as pd
import json
import torch
import preprocess
import random


def load_data_in_dictionary(path, specmax):
    pollen_types = get_sorted_json_file_names(path)
    data = DataDictionary(pollen_types)
    print('Classes: ', pollen_types)

    for pollen_type in pollen_types:
        raw_data = load_json(path, pollen_type)
        print('Class ', pollen_type)
        print('Particles before preprocessing: ', len(raw_data))

        for particle in raw_data:
            if np.max(particle["Spectrometer"]) > specmax:
                particle_data = preprocess_data(particle)
                if is_valid(particle_data):
                    data.append_particle(pollen_type, particle_data)

        data.print_length(pollen_type)
    data.permute()
    return data.get_dictionary()


def get_sorted_json_file_names(path):
    files = sorted(os.listdir(path))
    return [file.split(".")[0] for file in files if file.split('.')[1] == 'json']


class DataDictionary:
    def __init__(self, pollen_types):
        self.data_types = ['scattering image', 'lifetime', 'spectrum', 'size', 'lifetime weights']
        self.pollen_types = pollen_types
        self.dictionary = self.create_dictionary()

    def create_dictionary(self):
        dictionary = {key: {} for key in self.data_types}
        for data_type in self.data_types:
            for pollen_type in self.pollen_types:
                dictionary[data_type][pollen_type] = []
        dictionary['classes'] = self.pollen_types
        return dictionary

    def append_particle(self, pollen_type, particle_data):
        self.dictionary['scattering image'][pollen_type].append(particle_data['scat'])
        self.dictionary['size'][pollen_type].append(particle_data['size'])
        self.dictionary['lifetime'][pollen_type].append(particle_data['life'][0])
        self.dictionary['spectrum'][pollen_type].append(particle_data['spec'])
        self.dictionary['lifetime weights'][pollen_type].append(particle_data['life'][1])

    def permute(self):
        random.seed(0)
        for data_type in self.data_types:
            for pollen_type in self.pollen_types:
                list_to_permute = self.dictionary[data_type][pollen_type]
                self.dictionary[data_type][pollen_type] = random.sample(list_to_permute, len(list_to_permute))

    def print_length(self, pollen_type):
        print('Particles after preprocessing: ', len(self.dictionary['size'][pollen_type]))

    def get_dictionary(self):
        return self.dictionary


def load_json(path, file_name):
    file_path = path + '/' + file_name + '.json'
    return json.loads(open(file_path).read())


def preprocess_data(particle):
    scat = preprocess.normalize_image(particle, cut=60, normalize=False, smooth=True)
    life = preprocess.normalize_lifitime(particle, normalize=True)
    spec = preprocess.spec_correction(particle, normalize=True)
    size = preprocess.size_particle(particle)
    return {'scat': scat, 'life': life, 'spec': spec, 'size': size}


def is_valid(data):
    return data['scat'] is not None and data['spec'] is not None and data['life'] is not None


def save_as_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
    
def rearrange_data_dict(data):
    new_data = {}
    labels = []
    for d, data_type in enumerate(data):
        if data_type != 'classes':
            new_data[data_type] = []
            for i, pollen_type in enumerate(data[data_type]):
                new_data[data_type] += data[data_type][pollen_type]
                if d == 0:
                    labels += [i for _ in range(len(data[data_type][pollen_type]))]
    for d, data_type in enumerate(new_data):
        new_data[data_type] = np.array(new_data[data_type])
    labels = np.array(labels)
    
    return new_data, labels


def shuffle(tensor, labels):
    random.seed(0)
    ind = list(range(len(tensor['size'])))
    random.shuffle(ind)

    for data_type in tensor:
        tensor[data_type] = [tensor[data_type][i] for i in ind]
    labels = [labels[i] for i in ind]

    return tensor, labels


def get_tensors(data):
    tensor = {}
    for data_type in data:
        if data_type != 'classes':
            tensor[data_type] = create_tensor(data[data_type])
    return tensor


def create_tensor(data):
    dimension = len(data[0].shape)
    if dimension == 1:
        tensor = torch.as_tensor(data)
    elif dimension == 0:
        tensor = torch.as_tensor(data).unsqueeze_(0).permute(1, 0)
    else:
        tensor = torch.as_tensor(data).unsqueeze_(0).permute(1, 0, 2, 3)
    return tensor


def rearrange_data_for_TE(data):
    tensor = {}
    for dt in data:
        if dt in ["scattering image", "spectrum", "lifetime"]:
            tensor[dt] = torch.squeeze(data[dt]).permute(1, 0, 2)
        else:
            tensor[dt] = data[dt]

        if dt == 'scattering image' or dt == 'lifetime':
            tensor[dt] = torch.transpose(tensor[dt], 0, 2)
    return tensor



def get_hourly_data(path_to_data):
    data_tensor = load_pickle_data(path_to_data)
    hirst = pd.read_excel("../files/hirst hourly.xlsx")
    data, y_true, timestamps = [], [], []

    for ts in data_tensor:
        if int(ts) in list(hirst['datetime'].values) and len(data_tensor[ts]['size']) > 1:
            ind = list(hirst['datetime'].values).index(int(ts))
            y_true.append(list(hirst.iloc[ind, 1:].values))
            data.append(data_tensor[ts])
            timestamps.append(int(ts))

    return {'data': data, 'labels': y_true, 'timestamps': timestamps}


def shuffle_hourly_data(data):
    tensor, labels, timestamps = data['data'], data['labels'], data['timestamps']
    ind = list(range(len(timestamps)))
    random.seed(0)
    random.shuffle(ind)
    new_tensor = [tensor[i] for i in ind]
    new_labels = [labels[i] for i in ind]
    new_timestamps = [timestamps[i] for i in ind]
    return {'data': new_tensor, 'labels': new_labels, 'timestamps': new_timestamps}


def split_data_by_year(data, year):

    start = int(str(year) + '000000')
    end = int(str(year) + '500000')

    data_year, data_other = {}, {}
    for key in data:
        data_year[key] = [element for i,element in enumerate(data[key]) if start<data['timestamps'][i]<end]
        data_other[key] = [element for i,element in enumerate(data[key]) if not start<data['timestamps'][i]<end]

    return data_year, data_other


def get_daily_data(data, pollen_ind):
    daily_timestamps = np.array([int(str(ts)[:-2]) for ts in data['timestamps']])
    daily_data = {'data': [], 'labels': [], 'timestamps': [], 'num_hours_in_day': []}

    unique_daily_timestamps = np.unique(daily_timestamps).tolist()
    for ts in unique_daily_timestamps:
        temp_data_dict = {k: torch.cat([data['data'][i][k] for i, t in enumerate(daily_timestamps) if t == ts]) for k in data['data'][0].keys()}
        temp_labels_list = [data['labels'][i][pollen_ind] for i, t in enumerate(daily_timestamps) if t == ts]

        daily_data['data'].append(temp_data_dict)
        daily_data['labels'].append(np.sum(temp_labels_list))
        daily_data['timestamps'].append(ts)
        daily_data['num_hours_in_day'].append(len(temp_labels_list))

    return daily_data


def split_data_by_season(data, pollen_ind, thr):

    data_season = {'data': [], 'labels': [], 'timestamps': []}
    data_outofseason = {'data': [], 'labels': [], 'timestamps': []}
    for i, label in enumerate(data['labels']):
        if label[pollen_ind] > thr:
            data_season['data'].append(data['data'][i])
            data_season['labels'].append(data['labels'][i][pollen_ind])
            data_season['timestamps'].append(data['timestamps'][i])
        else:
            data_outofseason['data'].append(data['data'][i])
            data_outofseason['labels'].append(data['labels'][i][pollen_ind])
            data_outofseason['timestamps'].append(data['timestamps'][i])

    return data_season, data_outofseason


def split_data_by_indices(data, indices):
    tensor, labels, timestamps = data['data'], data['labels'], data['timestamps']
    X = [tensor[k] for k in indices]
    y = [labels[k] for k in indices]
    ts = [timestamps[k] for k in indices]
    return {'data': X, 'labels': y, 'timestamps': ts}


