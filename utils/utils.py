import os
import numpy as np

from dataset.dataset import *
from model import *

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def get_split_dataset(dataset_dict, split_number):
    dataset_path = dataset_dict['path']
    dataset_list = list()
    for i in range(split_number):
        dataset_dict['path'] = dataset_path + f'_{split_number}/{i}'
        dataset_list.append(get_dataset(dataset_dict))
    dataset_dict['path'] = dataset_path
    return dataset_list

def get_dataset(dataset_dict):
    name = dataset_dict['name']
    if name == 'Pair':
        return PairDataset(dataset_dict)
    else:
        raise ValueError(("No dataset " + name + " is found."))

def get_model(model_dict, dataset):
    name = model_dict['name']
    try:
        model_class = globals()[name]
        return model_class(model_dict, dataset)
    except KeyError:
        raise ValueError(f"No model {name} is found.")

def custom_topk(tensor, k):
    _, sorted_indices = torch.sort(tensor, dim=0, stable=True, descending=True)
    topk_indices = sorted_indices[:k]
    return _, topk_indices

class EarlyStopper_base(object):
    def __init__(self, num_trials, save_path, client_num):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path
        self.accuracy_list = []
        self.client_num = client_num
        self.save_path = save_path + '/best_model/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def is_continuable(self, model, accuracy, all_accuracy = None):
        if all_accuracy is not None:
            self.accuracy_list.append(all_accuracy)
        else:
            self.accuracy_list.append(accuracy)
        import pickle
        with open(self.save_path + '/acc_list', 'wb') as f:
            pickle.dump(self.accuracy_list, f)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            self.model_save(model)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

    def model_save(self, model_list):
        for i in range(self.client_num):
            torch.save(model_list[i], self.save_path + '/best_client_model_' + str(i) + '.pth')

    def model_load(self, model_list):
        for i in range(self.client_num):
            model_list[i] = torch.load(self.save_path + '/best_client_model_' + str(i) + '.pth', weights_only=False)
        return model_list
