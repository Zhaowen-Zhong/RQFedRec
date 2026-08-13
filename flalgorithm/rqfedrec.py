import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.logger import *
from utils.metric import *
from utils.utils import *


class RQFedRec(object):
    def __init__(self, config):
        self.build_logger(config)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Use device: {self.device}")

        self.build_dataset(config)
        self.build_model(config)
        self.build_global_config(config)

    def build_logger(self, config):
        exp_name = config.get("exp", "name")
        exp_output = config.get("exp", "output")
        self.logger, self.exp_save_name = logger(exp_name, exp_output)
        self.logger.info(f"logger is saved in: {self.exp_save_name}")
        set_seeds(config.getint("exp", "random_seed"))
        with open(self.exp_save_name + '/config.ini', 'w') as configfile:
            config.write(configfile)
        self.client_model_path = self.exp_save_name + '/clients/'
        os.mkdir(self.client_model_path)

    def build_dataset(self, config):
        dataset_dict = config['dataset']
        batch_size = config.getint("train", "batch_size")

        self.client_num = config.getint("dataset", "client_num")
        self.cluster_num = config.getint("train", "cluster_num")
        self.sid_length = config.getint("model", "sid_length")
        self.initial_sid = config.get("train", "initial_sid", fallback="semantic")

        self.dataset_list = get_split_dataset(dataset_dict, self.client_num)
        self.loader_list = [DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True) for dataset in self.dataset_list]
        self.generate_codebook(dataset_dict)

    def fit_codebook(self, residual):
        size_min = min(len(residual) // (self.cluster_num * 2), 50)
        size_max = size_min * 4

        from k_means_constrained import KMeansConstrained

        clf = KMeansConstrained(n_clusters=self.cluster_num, size_min=size_min, size_max=size_max, max_iter=10, n_init=10, n_jobs=10, verbose=False, random_state=42)
        clf.fit(residual)
        return clf.labels_, clf.cluster_centers_

    def generate_codebook(self, dataset_dict):
        dataset_path = dataset_dict['path'] + f'_{self.client_num}/'
        residual = np.load(dataset_path + 'item_embeddings.npy')

        if self.initial_sid == "random":
            rng = np.random.default_rng(42)
            self.sids = torch.tensor(rng.integers(0, self.cluster_num, size=(len(residual), self.sid_length)), dtype=torch.long)
            return

        sid = []
        for _ in range(self.sid_length):
            labels, cluster_centers = self.fit_codebook(residual)
            sid.append(labels)
            residual = residual - cluster_centers[labels]
        self.sids = torch.tensor(np.column_stack(sid), dtype=torch.long)

    def build_model(self, config):
        model_dict = config['model']
        self.early_stop_num = config.getint("train", "early_stop_num")
        self.lr = config.getfloat("train", "lr")
        self.weight_decay = config.getfloat("train", "weight_decay")

        self.client_model = []
        self.teacher_optimizer = []
        self.student_optimizer = []
        for i in range(self.client_num):
            model = get_model(model_dict, self.dataset_list[i])
            model.init_codebook(self.sids, self.cluster_num)
            self.client_model.append(model)
            self.teacher_optimizer.append(torch.optim.Adam(params=model.parameters(), lr=self.lr, weight_decay=self.weight_decay))
            self.student_optimizer.append(torch.optim.Adam(params=model.parameters(), lr=self.lr, weight_decay=self.weight_decay))

        self.server_model = get_model(model_dict, self.dataset_list[0])
        self.server_model.init_codebook(self.sids, self.cluster_num)
        self.server_model = self.server_model.to(self.device)
        self.server_paramname = [k for k in self.server_model.state_dict() if k.startswith('codebook_embeddings.')]
        self.download_paramname = self.server_paramname + ['sid']
        self.early_stopper = EarlyStopper_base(num_trials=self.early_stop_num, save_path=self.exp_save_name, client_num=self.client_num)

    def build_global_config(self, config):
        self.epoch = config.getint("train", "epoch")
        self.update_frequence = config.getint("train", "update_frequence")
        self.teacher_epoch = config.getint("train", "teacher_epoch")
        self.student_epoch = config.getint("train", "student_epoch")
        self.sid_update_interval = config.getint("train", "sid_update_interval", fallback=10)

    def client_init(self):
        for i in range(self.client_num):
            private_state = {k: self.server_model.state_dict()[k] for k in self.download_paramname}
            self.client_model[i].load_state_dict(private_state, strict=False)

    def train_client_model(self):
        for i in range(self.client_num):
            self.client_model[i] = self.client_model[i].to(self.device)
            self.client_model[i].train()
            for _ in range(self.update_frequence):
                for _ in range(self.teacher_epoch):
                    self.dataset_list[i].UniformSample()
                    for batch_users, batch_items, batch_ratings in self.loader_list[i]:
                        batch_users, batch_items, batch_ratings = batch_users.to(self.device), batch_items.to(self.device), batch_ratings.to(self.device)
                        loss = self.client_model[i].teacher_loss(batch_users, batch_items, batch_ratings)
                        self.teacher_optimizer[i].zero_grad()
                        loss.backward()
                        self.teacher_optimizer[i].step()

                for _ in range(self.student_epoch):
                    self.dataset_list[i].UniformSample()
                    for batch_users, batch_items, batch_ratings in self.loader_list[i]:
                        batch_users, batch_items, batch_ratings = batch_users.to(self.device), batch_items.to(self.device), batch_ratings.to(self.device)
                        loss = self.client_model[i].student_loss(batch_users, batch_items, batch_ratings)
                        self.student_optimizer[i].zero_grad()
                        loss.backward()
                        self.student_optimizer[i].step()
            self.client_model[i] = self.client_model[i].to('cpu')

    def client_upload(self):
        datasize_sum = 0
        aggstate = {k: 0.0 for k in self.server_paramname}
        for i in range(self.client_num):
            datasize = self.dataset_list[i].__len__()
            datasize_sum += datasize
            self.client_model[i] = self.client_model[i].to(self.device)
            for k in self.server_paramname:
                aggstate[k] = aggstate[k] + self.client_model[i].state_dict()[k] * datasize
            self.client_model[i] = self.client_model[i].to("cpu")

        for k in self.server_paramname:
            aggstate[k] = aggstate[k] / datasize_sum
        self.server_model.load_state_dict(aggstate, strict=False)

    def client_download(self):
        for i in range(self.client_num):
            private_state = {k: self.server_model.state_dict()[k] for k in self.download_paramname}
            self.client_model[i].load_state_dict(private_state, strict=False)
            self.client_model[i] = self.client_model[i].to(self.device)
            self.client_model[i].teacher_update()
            self.client_model[i] = self.client_model[i].to('cpu')

    def reset_codebook_optimizer_state(self):
        for model, optimizer in zip(self.client_model, self.student_optimizer):
            for codebook in model.codebook_embeddings:
                optimizer.state.pop(codebook.weight, None)

    @torch.no_grad()
    def update_data(self):
        if self.epoch_i < self.sid_update_interval:
            return False
        if self.epoch_i % self.sid_update_interval != 0:
            return False

        all_items = torch.arange(self.server_model.num_items, device=self.device)
        residual = self.server_model.get_item_emb(all_items).detach().cpu().numpy()
        new_sid = []

        for sid_col in range(self.sids.shape[1]):
            labels, cluster_centers_np = self.fit_codebook(residual)
            cluster_centers = torch.tensor(cluster_centers_np, device=self.device, dtype=self.server_model.codebook_embeddings[sid_col].weight.dtype)
            labels = torch.cdist(torch.tensor(residual, device=self.device, dtype=cluster_centers.dtype), cluster_centers).argmin(dim=1)
            self.server_model.codebook_embeddings[sid_col].weight.data.copy_(cluster_centers)
            new_sid.append(labels.cpu().numpy())
            residual = residual - cluster_centers[labels].cpu().numpy()

        old_sid = self.server_model.sid.detach().clone()
        self.server_model.sid.copy_(torch.tensor(np.column_stack(new_sid), device=self.device, dtype=torch.long))
        self.server_model.teacher_update()
        self.reset_codebook_optimizer_state()
        return True

    def evaluate_valid(self):
        predictions = []
        labels = []
        for i in range(self.client_num):
            self.client_model[i].eval()
            self.client_model[i] = self.client_model[i].to(self.device)
            for user, data in enumerate(self.dataset_list[i].valid):
                this_pred = self.client_model[i].getUsersRating(user, data)
                scores = torch.full((self.dataset_list[i].item_num,), -1.0).to(self.device)
                scores[data] = this_pred
                trainset = torch.tensor(self.dataset_list[i].trainset[user] > 0)
                scores[trainset] = -1.0
                prediction = custom_topk(scores, k=10)[1].cpu().numpy()
                predictions.append(prediction)
                labels.append(self.dataset_list[i].validlabel[user])
            self.client_model[i] = self.client_model[i].to('cpu')
        return calculate_all(labels, predictions, [10])

    def evaluate_test(self):
        predictions = []
        labels = []
        for i in range(self.client_num):
            self.client_model[i].eval()
            self.client_model[i] = self.client_model[i].to(self.device)
            for user, data in enumerate(self.dataset_list[i].test):
                this_pred = self.client_model[i].getUsersRating(user, data)
                scores = torch.full((self.dataset_list[i].item_num,), -1.0).to(self.device)
                scores[data] = this_pred
                trainset = torch.tensor(self.dataset_list[i].trainset[user] > 0)
                validset = torch.tensor(self.dataset_list[i].validset[user] > 0)
                scores[trainset] = -1.0
                scores[validset] = -1.0
                prediction = custom_topk(scores, k=10)[1].cpu().numpy()
                predictions.append(prediction)
                labels.append(self.dataset_list[i].testlabel[user])
            self.client_model[i] = self.client_model[i].to('cpu')
        return calculate_all(labels, predictions, [10])

    def fit(self):
        for epoch_i in range(self.epoch):
            start_time = time.time()
            self.epoch_i = epoch_i

            if epoch_i == 0:
                self.client_init()
                self.logger.info(f"Finish init client model in {(time.time() - start_time):.2f}s")

            self.train_client_model()
            self.logger.info(f"Finish client train in {(time.time() - start_time):.2f}s")

            self.client_upload()
            self.logger.info(f"Finish upload in {(time.time() - start_time):.2f}s")

            if epoch_i > 0:
                self.update_data()

            self.client_download()
            self.logger.info(f"Finish download in {(time.time() - start_time):.2f}s")

            valid_metric = self.evaluate_valid()
            self.logger.info(f"Finish valid valid set epoch {epoch_i} in {(time.time() - start_time):.2f}s, recall: {valid_metric[1]}, mrr: {valid_metric[3]}, ndcg: {valid_metric[4]}.")

            if epoch_i < self.sid_update_interval:
                self.logger.info(f"Skip early stop before warm epoch {self.sid_update_interval}.")
                continue

            if not self.early_stopper.is_continuable(self.client_model, valid_metric[1][0], all_accuracy=valid_metric):
                self.logger.info(f'early stop at epoch {epoch_i}')
                self.logger.info(f'validation: best recall: {self.early_stopper.best_accuracy}')
                self.client_model = self.early_stopper.model_load(self.client_model)
                break

        test_metric = self.evaluate_test()
        self.logger.info(f"recall: {test_metric[1]}, mrr: {test_metric[3]}, ndcg: {test_metric[4]}.")
