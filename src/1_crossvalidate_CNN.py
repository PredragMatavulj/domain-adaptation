import Libraries.prepare_data as prepare_data
import Libraries.architecture as architecture
from Libraries.trainer import Trainer
from sklearn.model_selection import StratifiedKFold
import torch


data = prepare_data.load_pickle_data('../../pollen/data/data_ver18.pckl')
data_rearranged, labels = prepare_data.rearrange_data_dict(data)
classes = data['classes']

skf_outter = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
for fold_outter, (tr_index, test_index) in enumerate(skf_outter.split(data_rearranged['scattering image'], labels)):
    
    X_tr, X_test = {}, {}
    for data_type in data_rearranged:
        X_tr[data_type], X_test[data_type] = data_rearranged[data_type][tr_index], data_rearranged[data_type][test_index]
    y_tr, y_test = labels[tr_index], labels[test_index]
    
    skf_inner = StratifiedKFold(n_splits=10)
    for fold_inner, (train_index, val_index) in enumerate(skf_inner.split(X_tr['scattering image'], y_tr)):
        
        X_train, X_val = {}, {}
        for data_type in X_tr:
            X_train[data_type], X_val[data_type] = X_tr[data_type][train_index], X_tr[data_type][val_index]
        y_train, y_val = y_tr[train_index], y_tr[val_index]

        train, train_labels = prepare_data.shuffle(X_train, y_train)
        val, val_labels =  prepare_data.shuffle(X_val, y_val)
    
        model = architecture.Cnn(len(classes))
        model_name = f'test{fold_outter}_val{fold_inner}'

        train_test_info = {
            'model_name': model_name,
            'gpu': 2,
            'batch_size': 500,
            'optimizer': torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9),
            'loss': torch.nn.NLLLoss(),
            'model_to_train': model,
            'train_tensor': prepare_data.get_tensors(train),
            'train_labels': torch.LongTensor(train_labels),
            'val_tensor': prepare_data.get_tensors(val),
            'val_labels': torch.LongTensor(val_labels),
            'test_tensor': prepare_data.get_tensors(X_test),
            'test_labels': y_test,
            'classes': classes,
        }
        tr = Trainer(train_test_info)
        print(model_name)
        tr.train()
        tr.test()
