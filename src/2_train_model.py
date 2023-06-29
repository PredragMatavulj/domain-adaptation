import Libraries.prepare_data as prepare_data
import Libraries.architecture as architecture
from Libraries.trainer import Trainer
import torch


data = prepare_data.load_pickle_data('../../pollen/rapid_e/data/data_ver18.pckl')
data_rearranged, labels = prepare_data.rearrange_data_dict(data)
classes = data['classes']

train, train_labels = prepare_data.shuffle(data_rearranged, labels)

model = architecture.Cnn(len(classes))
train_test_info = {
    'model_name': 'final_model',
    'gpu': 0,
    'batch_size': 500,
    'optimizer': torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9),
    'loss': torch.nn.NLLLoss(),
    'model_to_train': model,
    'train_tensor': prepare_data.get_tensors(train),
    'train_labels': torch.LongTensor(train_labels),
    'val_tensor': None,
    'val_labels': None,
    'test_tensor': None,
    'test_labels': None,
    'classes': classes,
    'train_num_of_epochs': 0 # if 0, train until converges, else train for specified num_of_epochs
}
tr = Trainer(train_test_info)
tr.train('final')
