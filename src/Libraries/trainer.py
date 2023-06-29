import numpy as np
import random
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib
import random
import prepare_data
from scipy.stats import spearmanr
import os


class Trainer:
    def __init__(self, train_info):
        self.gpu = train_info['gpu']
        self.model = train_info['model_to_train']
        self.batch_size = train_info['batch_size']
        self.optimizer = train_info['optimizer']
        self.loss_f = train_info['loss']
        self.train_tensor = train_info['train_tensor']
        self.train_labels = train_info['train_labels']
        self.val_tensor = train_info['val_tensor']
        self.val_labels = train_info['val_labels']
        self.test_tensor = train_info['test_tensor']
        self.test_labels = train_info['test_labels']
        self.classes = train_info['classes']
        self.model_name = train_info['model_name']
        self.train_losses, self.val_losses = [], []
        self.epoch, self.best_epoch,self.early_stopping = 0, 0, 0
        self.path_to_model = train_info['path_to_models']

        self.train_tensor = prepare_data.rearrange_data_for_TE(self.train_tensor)
        self.val_tensor = prepare_data.rearrange_data_for_TE(self.val_tensor)
        self.test_tensor = prepare_data.rearrange_data_for_TE(self.test_tensor)


    def train(self, mode = 'validate'):
        try:
            while True:
                self.transfer_to_cuda(mode)
                train_loss = self.fit()
                if mode == 'validate':
                    val_loss = self.validate()
                    print("- Train Loss = %a, Val Loss = %a" % (train_loss, val_loss))

                    if self.val_losses and val_loss < np.min(self.val_losses):
                        self.save_model()
                        self.best_epoch = self.epoch

                    #self.plot_losses()
                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)
                else:
                    print('Epoch: ', self.epoch, '; loss: ', train_loss)

                self.epoch += 1

                self.weight_decay()
                if self.early_stopping > 6:
                    self.save_model()
                    print("Training has finished.")
                    break
        except KeyboardInterrupt:
            print("Training stopped by user.")


    def transfer_to_cuda(self, mode = 'validate'):
        if torch.cuda.is_available():
            self.model = self.model.cuda(self.gpu)
            self.loss_f = self.loss_f.cuda(self.gpu)
            self.train_labels = self.train_labels.cuda(self.gpu)
            if mode == 'validate':
                self.val_labels = self.val_labels.cuda(self.gpu)
        else:
            print('Cuda not available!')


    def fit(self):
        print("Epoch %a: Train[" % self.epoch, end="")
        batch_losses = []
        self.model.train()
        num_of_batches = int(np.floor(len(self.train_tensor['size'])/self.batch_size))
        for batch in range(num_of_batches):
            self.optimizer.zero_grad()

            model_input = self.get_model_input('train', batch)
            labels = self.train_labels[batch*self.batch_size:batch*self.batch_size + self.batch_size]

            model_output = self.model(model_input)
            loss = self.loss_f(model_output, labels)
            loss.backward()
            self.optimizer.step()
            batch_losses.append(loss.data.cpu().numpy())
            print("*", end="")
        return np.mean(batch_losses)


    def get_model_input(self, data, batch):
        if data == 'train':
            tensor = self.train_tensor
        else:
            tensor = self.val_tensor

        model_input = {}
        for data_type in tensor:
            if data_type != 'classes':
                if data_type == 'scattering image' or data_type == 'lifetime' or data_type == 'spectrum':
                    model_input[data_type] = tensor[data_type][:, batch*self.batch_size:batch*self.batch_size + self.batch_size, :]
                    model_input[data_type] = model_input[data_type].cuda(self.gpu)
                else:
                    model_input[data_type] = tensor[data_type][batch*self.batch_size:batch*self.batch_size + self.batch_size]
                    model_input[data_type] = model_input[data_type].cuda(self.gpu)

        return model_input


    def validate(self):
        print("] Validation[", end="")
        batch_losses = []
        self.model.eval()
        num_of_batches = int(np.floor(len(self.val_tensor['size'])/self.batch_size))
        for batch in range(num_of_batches):

            model_input = self.get_model_input('val', batch)
            labels = self.val_labels[batch*self.batch_size:batch*self.batch_size + self.batch_size]

            model_output = self.model(model_input)
            loss = self.loss_f(model_output, labels)
            batch_losses.append(loss.data.cpu().numpy())
            print("*", end="")
        print("] ", end="")
        return np.mean(batch_losses)


    def plot_losses(self):
        plt.figure(figsize=(10, 8))
        plt.plot(range(self.epoch), self.train_losses)
        plt.plot(range(self.epoch), self.val_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Train", "Validation"], fontsize=20)
        matplotlib.rcParams.update({'font.size': 20})
        #plt.savefig(self.path_to_model + 'err/' + self.model_name + '_err.png')
        plt.show()


    def is_overfitting(self, num_of_losses):
        return np.min(self.val_losses[-num_of_losses:]) > np.min(self.val_losses)


    def reduce_learning_rate(self, lr):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)


    def weight_decay(self):
        if self.early_stopping == 0:
            if self.epoch == 75:
                self.reduce_learning_rate(0.01)
                self.early_stopping = 1
        elif self.early_stopping == 1:
            if self.epoch == 125:
                self.reduce_learning_rate(0.0001)
                self.early_stopping = 2
        elif 2 <= self.early_stopping <= 6:
            self.early_stopping += 1


    def save_model(self):
        torch.save(self.model.state_dict(), self.path_to_model + self.model_name + '.pth')


    def test(self):
        self.model.load_state_dict(torch.load(self.path_to_model + self.model_name + '.pth', map_location=lambda storage, loc: storage))
        self.model.cpu()
        self.model.eval()

        outputs = self.model(self.test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        y_pred = predicted.cpu().numpy()

        self.save_metrics(y_pred)
        #self.plot_confusion_matrix(y_pred)


    def save_metrics(self, y_pred):
        y_true = self.test_labels

        metrics = {
            'model_saved_at_epoch' : self.best_epoch,
            'confusion_matrix' : confusion_matrix(y_true, y_pred),
            'f1_macro' : f1_score(y_true, y_pred, average="macro"),
            'precision_macro' : precision_score(y_true, y_pred, average="macro"),
            'recall_macro' : recall_score(y_true, y_pred, average="macro"),
            'f1_weighted' : f1_score(y_true, y_pred, average="weighted"),
            'precision_weighted' : precision_score(y_true, y_pred, average="weighted"),
            'recall_weighted' : recall_score(y_true, y_pred, average="weighted"),
            'accuracy' : accuracy_score(y_true, y_pred),
            'train_losses': self.train_losses,
            'val_losses' : self.val_losses
        }
        prepare_data.save_as_pickle(metrics, self.path_to_model + 'metrics/' + self.model_name + '_metrics')




class ReTrainer:
    def __init__(self, train_info):
        self.data_combination = train_info['data_combination']
        self.device = train_info['device']
        self.batch_size = train_info['batch_size']
        self.model = train_info['model_to_train'].to(self.device)
        self.optimizer = train_info['optimizer']
        self.loss_f = train_info['loss'].to(self.device)
        self.train = train_info['train']
        self.val = train_info['val']
        self.test_d = train_info['test_season']
        self.train_out = train_info['train_outofseason']
        self.test_out = train_info['test_outofseason']
        self.pollen_ind = train_info['pollen_ind']
        self.model_name = train_info['model_name']
        self.train_losses, self.val_losses = [], []
        self.epoch, self.best_epoch, self.early_stopping = 0, 0, 0
        self.model_path = os.path.join(train_info['path_to_models'], self.model_name + '.pth')


    def train(self, mode='validate'):
        while True:
            print(f"Epoch {self.epoch}: ", end="")
            self.prepare_data()
            train_loss = self.fit()

            if mode == 'validate':
                val_loss = self.validate()
                print(f" - Train Loss = {train_loss}, Val Loss = {val_loss}")

                if not self.val_losses or val_loss < min(self.val_losses):
                    self.save_model()
                    self.best_epoch = self.epoch

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
            else:
                print(f"; Loss: {train_loss}")

            self.weight_decay(50)

            if self.early_stopping == 50:
                self.save_model()
                print("Training has finished.")
                break

            self.epoch += 1


    def prepare_data(self):
        if self.data_combination != '100_0':
            ind = list(range(len(self.train_out['data'])))

            if self.data_combination == '75_25':
                k=3
            elif self.data_combination == '50_50':
                k=1
            tr = np.int(np.floor(len(self.train['data'])/k))
            va = np.int(np.floor(len(self.val['data'])/k))

            new_ind = random.sample(ind, tr+va)
            train_ind = new_ind[:tr]
            val_ind = new_ind[tr:]

            train_tensor = self.train['data'] + [self.train_out['data'][i] for i in train_ind]
            train_labels = self.train['labels'] + [self.train_out['labels'][i] for i in train_ind]
            val_tensor = self.val['data'] + [self.train_out['data'][i] for i in val_ind]
            val_labels = self.val['labels'] + [self.train_out['labels'][i] for i in val_ind]

            #shuffle
            train_ind = list(range(len(train_tensor)))
            val_ind = list(range(len(val_tensor)))
            random.shuffle(train_ind)
            random.shuffle(val_ind)

            self.train_data = ([train_tensor[i] for i in train_ind], [train_labels[i] for i in train_ind])
            self.val_data = ([val_tensor[i] for i in val_ind], [val_labels[i] for i in val_ind])
        else:
            self.train_data = (self.train['data'], self.train['labels'])
            self.val_data = (self.val['data'], self.val['labels'])


    def fit(self):
        print("Train[", end="")
        batch_losses = []
        self.model.train()
        num_batches = int(np.ceil(len(self.train_data)/self.batch_size))

        for batch in range(num_batches):

            model_input, labels = self.get_model_input(self.train_data, batch)
            self.optimizer.zero_grad()
            model_output = self.model(model_input, self.pollen_ind)
            loss = self.loss_f(model_output.flatten(), labels)
            loss.backward()
            self.optimizer.step()
            batch_losses.append(loss.item())
            print("*", end="")
        print("]", end="")
        return np.mean(batch_losses)


    def get_model_input(self, data, batch):

        tensor, labels = data
        model_input = tensor[batch*self.batch_size : (batch+1)*self.batch_size]
        batch_labels = labels[batch*self.batch_size : (batch+1)*self.batch_size]

        with torch.no_grad():
            batch_labels = torch.tensor(batch_labels)
            model_input = [{k: v.to(self.device) for k, v in sample.items()} for sample in model_input]

        return model_input, self.normalize(batch_labels)


    def normalize(self, labels):
        return torch.nn.functional.normalize(labels.float(), p=1, dim=0)


    def validate(self):
        print("Validate[", end="")
        batch_losses = []
        self.model.eval()
        num_batches = int(np.ceil(len(self.val_data)/self.batch_size))

        for batch in range(num_batches):

            model_input, labels = self.get_model_input(self.val_data, batch)
            with torch.no_grad():
                model_output = self.model(model_input, self.pollen_ind)
            loss = self.loss_f(model_output.flatten(), labels)
            batch_losses.append(loss.item())
            print("*", end="")
        print("]", end="")
        return np.mean(batch_losses)


    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)


    def weight_decay(self, num_losses):
        if self.early_stopping == 0:
            if self.is_overfitting(num_losses):
                self.model.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage))
                self.optimizer.param_groups[0]['lr'] = 0.0001
                self.early_stopping = 1
        elif 1 <= self.early_stopping < 50:
            self.early_stopping += 1


    def is_overfitting(self, num_losses):
        recent_val_losses = self.val_losses[-num_losses:]
        return np.min(recent_val_losses) > np.min(self.val_losses)


    def test(self):
        self.model.eval()

        test_data = (self.test_d['data'], self.test_d['labels'])
        test_out_data = (self.test_out['data'], self.test_out['labels'])

        test_outputs, test_labels = self.compute_outputs(test_data)
        corr, p_value = spearmanr(test_outputs.flatten(), test_labels)
        mean_in_season = torch.mean(test_outputs)

        out_of_season_outputs, _ = self.compute_outputs(test_out_data)
        mean_out_of_season = torch.mean(out_of_season_outputs)

        model_info = {
            'model_weights': self.model.state_dict(),
            'model_saved_at_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses' : self.val_losses,
            'corr_in_season': corr,
            'pval_in_season': p_value,
            'mean_in_season': mean_in_season,
            'mean_out_of_season': mean_out_of_season,
            'timestamps': self.test_d['timestamps']
        }
        torch.save(model_info, self.model_path)


    def compute_outputs(self, data):
        outputs, labels = [], []
        num_batches = int(np.ceil(len(data)/self.batch_size))
        with torch.no_grad():
            for batch in range(num_batches):
                model_input, label = self.get_model_input(data, batch)
                model_output = self.model(model_input, self.pollen_ind)
                outputs.append(model_output)
                labels.append(label)
        return torch.cat(outputs), torch.cat(labels)

