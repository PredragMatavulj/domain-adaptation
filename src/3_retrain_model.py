import sys
sys.path.insert(1, 'Libraries')
import torch
import Libraries.architecture as architecture
from Libraries.prepare_data import (get_hourly_data, split_data_by_season, split_data_by_year, shuffle_hourly_data, split_data_by_indices, get_daily_data)
from Libraries.trainer import ReTrainer
from sklearn.model_selection import KFold


CLASSES = ['Acer', 'Alnus', 'Ambrosia', 'Artemisia', 'Betula', 'Broussonetia', 'Carpinus', 'Corylus', 
            'Fraxinus', 'Juglans', 'Morus', 'Pinaceae', 'Plantago', 'Platanus', 'Poaceae', 'Populus', 
            'Quercus', 'Rumex', 'Salix', 'Taxaceae', 'Tilia', 'Ulmus', 'Urticaceae', 'skrob', 'spore&plantmaterial']

GPU = 0
DEVICE = torch.device('cuda', GPU) if torch.cuda.is_available() else torch.device('cpu')
TEST_YEARS = [2019, 2020, 2021]
MODEL_PATH = '../models/before_retraining/final_model.pth'
DATA_PATH = '../data/tl'

BATCH_SIZE = 124
LEARNING_RATE = 0.01
MOMENTUM = 0.9

data = get_hourly_data(DATA_PATH)

for test_year in TEST_YEARS:
    data_test, data_tr = split_data_by_year(data, test_year)   
    data_tr = shuffle_hourly_data(data_tr)

    for pollen in CLASSES[:-2]:
        pollen_ind = CLASSES.index(pollen)

        test_season, test_outofseason = split_data_by_season(data_test, pollen_ind, 10)
        train_season, train_outofseason = split_data_by_season(data_tr, pollen_ind, 10)

        kf = KFold(n_splits=10)
        for fold, (train_ind, val_ind) in enumerate(kf.split(train_season['timestamps'], train_season['timestamps'])):
            data_train = split_data_by_indices(train_season, train_ind)
            data_val = split_data_by_indices(train_season, val_ind)

            for data_combination in ['100_0', '75_25', '50_50']:
                model = architecture.CnnTLone(len(CLASSES))
                with open(MODEL_PATH, 'rb') as f:
                    model.load_state_dict(torch.load(f, map_location=DEVICE))
                model_name = f"retrained_{pollen}_data_{data_combination}_test_{test_year}_val{fold}"

                optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
                loss = torch.nn.MSELoss()
                
                train_test_info = {
                    'data_combination': data_combination,
                    'model_name': model_name,
                    'device': DEVICE,
                    'batch_size': BATCH_SIZE,
                    'optimizer': optimizer,
                    'loss': loss,
                    'model_to_train': model,
                    'train': data_train,
                    'train_outofseason': train_outofseason,
                    'val': data_val,
                    'test_season': test_season,
                    'test_outofseason': test_outofseason,
                    'pollen_ind': pollen_ind,
                    'path_to_models': '../models/after/'
                }
                tr = ReTrainer(train_test_info)
                print(model_name)
                tr.retrain()
                tr.test()
