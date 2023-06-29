import Libraries.prepare_data as prepare_data
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn import preprocessing
import Libraries.architecture as architecture
import torch


params_list = ['Acer', 'Alnus', 'Ambrosia', 'Artemisia', 'Betula', 'Broussonetia', 'Carpinus', 'Corylus',
                    'Fraxinus', 'Juglans', 'Morus', 'Pinaceae', 'Plantago', 'Platanus', 'Poaceae', 'Populus',
                    'Quercus', 'Rumex', 'Salix', 'Taxaceae', 'Tilia', 'Ulmus', 'Urticaceae', 'skrob', 'spore&plantmaterial']


def save_model_output(path_to_data, model_name):
    data_tensor = prepare_data.load_pickle_data(path_to_data)
    predictions = get_model_predictions(model_name, data_tensor)
    daily_model = get_daily_values(predictions)
    daily_model.to_excel('../models/model_outputs/' + model_name + '.xlsx', index = False)


def get_model_predictions(model_name, data_tensor):
    model = architecture.Cnn(len(params_list))
    model.load_state_dict(torch.load('../models/' + model_name + '.pth', map_location='cpu'))
    model.eval()
    
    predictions = {}
    softmax = torch.nn.Softmax(dim=1)

    for ts in data_tensor:
        output = softmax(model(data_tensor[ts]))
        _, pred = torch.max(output.data, 1)
        predictions[ts] = [list(pred).count(cl) for cl in range(len(params_list))]

    return predictions


def get_daily_values(preds):

    aggregated_hourly = {}
    for ts in preds:
        concentrations = [preds[ts][i]*5.95 for i in range(len(preds[ts]))]
        if ts[:8] not in aggregated_hourly:
            aggregated_hourly[ts[:8]] = [concentrations]
        else:
            aggregated_hourly[ts[:8]].append(concentrations)

    daily_values = {}
    for ts in aggregated_hourly:
        daily_values[ts] = [ts, len(aggregated_hourly[ts])] + list(np.mean(aggregated_hourly[ts], 0))

    daily_model = pd.DataFrame(np.zeros((len(daily_values), len(params_list) + 2)))
    daily_model.columns = ['date', 'num_of_hours_in_average'] + params_list
    for i, ts in enumerate(daily_values):
        daily_model.iloc[i, :] = daily_values[ts]

    return daily_model



def save_correlation_distributions(thr, pollens, hirst_data, models_data):
    results = {k: [] for k in ['corrs_mean', 'corrs_std', 'p_val_mean']}

    for pollen in pollens:
        correlations, p_values = [], []

        if pollen in hirst_data.columns and pollen in models_data.columns:
            hirst_pollen = hirst_data[pollen]
            model_pollen = models_data[pollen]

            indices = np.where(hirst_pollen >= thr)[0]

            if indices.size > 0:
                ind_model, ind_hirst = [], []

                for ind in indices:
                    date = hirst_data.at[ind, 'date']
                    if date in model_pollen['date'].values:
                        ind_model.append(model_pollen['date'].tolist().index(date))
                        ind_hirst.append(ind)

                if ind_model:
                    model_mean, model_std = get_model_mean_and_std(models_data, pollen, ind_model)

                    for _ in range(10000):
                        hirst_new = get_new_hirst(hirst_pollen, ind_hirst)
                        model_new = get_new_model(model_mean, model_std)

                        correlation, p_value = spearmanr(hirst_new, model_new)
                        correlations.append(correlation)
                        p_values.append(p_value)
        else:
            print(pollen)

        if correlations:
            values_to_add = [np.mean(correlations), np.std(correlations), np.mean(p_values)]
        else:
            values_to_add = [np.nan, np.nan, np.nan]
        results = add_values(results, values_to_add)

    write_to_excel(results, pollens)


def add_values( results, values):
    for i, key in enumerate(results):
        results[key].append(values[i])


def get_hirst():
    path_to_hirst = '../files/hirst daily.xlsx'
    hirst = pd.read_excel(path_to_hirst)
    hirst = remove_calibrations_from_dataframe(hirst)
    return hirst


def remove_calibrations_from_dataframe(dataframe):
    rows = get_calibration_indices(dataframe)
    return dataframe.drop(rows).reset_index().drop('index', axis = 1)


def get_calibration_indices(dataframe):

    calibration_dates = ['20180417', '20180418', '20180419', '20180420', '20180423', '20180504', 
                        '20180510', '20180515', '20180519', '20180528', '20180607', '20180615', 
                        '20180630', '20180703', '20180716', '20180718', '20180807', '20180829', 
                        '20181012', '20181018', '20181029', '20181210', '20181211', '20190125', 
                        '20190205', '20190206', '20190207', '20190211', '20190218', '20190225', 
                        '20190311', '20190315', '20190319', '20190325', '20190326', '20190329', 
                        '20190404', '20190409', '20190415', '20190418', '20190422', '20190429', 
                        '20190503', '20190510', '20190524', '20190603', '20210220', '20210221', 
                        '20210222', '20210304', '20210318', '20210521', '20210524', '20210604', 
                        '20210614', '20210826']

    calibration_indices = []
    for datee in calibration_dates:
        dataframe_index = np.where(dataframe['date'] == datee)[0]
        if len(dataframe_index) > 0:
            calibration_indices.append(dataframe_index[0])
            
    return calibration_indices


def get_models():
    models = {}
    for outter_fold in range(10):
        for inner_fold in range(10):
            model_name = 'test' + str(outter_fold) + '_val' + str(inner_fold)
            path_to_model = '../models/model_outputs/' + model_name + '.xlsx'
            model = pd.read_excel(path_to_model)
            model = remove_calibrations_from_dataframe(model)
            models[model_name] = model
    return models


def get_model_mean_and_std(models, pollen, ind):
    models_norm = []
    for model_name in models:
        models_norm.append(preprocessing.StandardScaler().fit_transform(models[model_name][pollen][ind].values.reshape(-1, 1)).flatten())
    return np.mean(models_norm, 0), np.std(models_norm, 0)


def get_new_hirst( hirst, ind):
    hirst_new = []
    for val in list(hirst[ind].values):
        hirst_new = add_hirst_value(val, hirst_new)
    return hirst_new


def add_hirst_value(val_to_add, hirst_new):
    if val_to_add < 30:
        hirst_new.append(val_to_add + np.random.normal(0, 1)*val_to_add*0.3)
    elif val_to_add >= 30 and val_to_add < 300:
        hirst_new.append(val_to_add + np.random.normal(0, 1)*val_to_add*0.2)
    else:
        hirst_new.append(val_to_add + np.random.normal(0, 1)*val_to_add*0.1)
    return hirst_new


def get_new_model(model_mean, model_std):
    model_new = []
    for mean, std in zip(model_mean, model_std):
        model_new.append(mean + np.random.normal(0, 1)*std)
    return model_new


def write_to_excel(dict, pollens):
    dfs = {}
    for key in dict:
        dfs[key] = create_dataframe(dict[key], [0], pollens)

    with pd.ExcelWriter('../files/correlations from distribution/' + key + '.xlsx') as writer:
        for key in dfs:
            dfs[key].to_excel(writer, sheet_name = key)


def create_dataframe(matrix, row_names, column_names):
    matrix = np.array(matrix)
    if len(matrix.shape) == 1:
        df = pd.DataFrame(np.expand_dims(matrix, 0))
    else:
        df = pd.DataFrame(matrix)
    df.index = row_names
    df.columns = column_names
    d = df.replace(0, np.NaN)
    df['MEAN'] = d.mean(1)
    return df




def main():
    data_path = '../data/tl'
    for outter_fold in range(10):            
        for inner_fold in range(10):
            model_name = f'test{outter_fold}_val{inner_fold}'
            save_model_output(data_path, model_name)

    save_correlation_distributions(10, params_list[:-2], get_hirst(), get_models())


if __name__ == "__main__":
    main()