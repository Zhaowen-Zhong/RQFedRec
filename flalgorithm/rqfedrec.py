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

        # 新增加FedCodeFair需要的配置
        """
        meta_lr是元学习内循环（支持集适应）的学习率。
        对应方案：元学习双层循环中的内循环。
        因为离散码不可以直接求导，所以这里的学习率主要用于将码分配来放宽成概率分布后的连续参数更新。
        实现目标：控制新用户面对少量支持集时的快速适应速度。
        """
        self.meta_lr=config.getfloat("train","meta_lr",fallback=0.01)
        
        """
        meta_steps是元学习内循环的梯度下降步数。
        对应方案：用支持集适应、接着查询集评估、元梯度的 1~5 步梯度回传。
        实现目标：决定模型在本地新任务上适应多少步，步数越多适应可能越好，但计算开销越大。
        """
        self.meta_steps = config.getint("train", "meta_steps", fallback=5)

        """
        sem_threshold: 码共现图中判定“语义邻居”的共享码数量阈值。
        对应方案：规定共享 H-1 个码作为正样本的标准。
        实现目标：例如设置为2，意味着两个用户至少要在 (h,k) 层级上共享2个相同的码，才认为他们是语义邻居，避免噪声连接。
        """
        self.sem_threshold = config.getint("train", "sem_threshold", fallback=2)

        """
        trim_ratio: Trimmed-Mean 鲁棒聚合的裁剪比例。
        对应方案：服务器聚合时“每个参数维度去掉最大和最小的数值，剩余取平均”。
        实现目标：防御投毒攻击。0.1表示去掉参数分布中最大10%和最小10%的值,剩下求平均。
        """
        self.trim_ratio = config.getfloat("train", "trim_ratio", fallback=0.1)
        
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
        
        # 新增数据流容器
        self.client_sids = [] # 存储客户端上传的码序列         
        self.semantic_neighbors = [] # 码共现图构建的语义邻居  
        self.behavior_neighbors = [] # F2PGNN行为邻居预留  
        self.H_u_list = [self.sid_length] * self.client_num # 预留：每个用户的隐私上传参数
        
    
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

    # 客户端本地训练，融合元学习
    def train_client_model(self):
        for i in range(self.client_num):
            self.client_model[i] = self.client_model[i].to(self.device)
            self.client_model[i].train()
            
            # 原有逻辑：基础teacher/student训练
            for _ in range(self.update_frequence):
                # 1. teacher模型训练
                for _ in range(self.teacher_epoch):
                    self.dataset_list[i].UniformSample()
                    for batch_users, batch_items, batch_ratings in self.loader_list[i]:
                        batch_users, batch_items, batch_ratings = batch_users.to(self.device), batch_items.to(self.device), batch_ratings.to(self.device)
                        loss = self.client_model[i].teacher_loss(batch_users, batch_items, batch_ratings)
                        self.teacher_optimizer[i].zero_grad()
                        loss.backward()
                        self.teacher_optimizer[i].step()

                # 2. student模型训练
                for _ in range(self.student_epoch):
                    self.dataset_list[i].UniformSample()
                    for batch_users, batch_items, batch_ratings in self.loader_list[i]:
                        batch_users, batch_items, batch_ratings = batch_users.to(self.device), batch_items.to(self.device), batch_ratings.to(self.device)
                        loss = self.client_model[i].student_loss(batch_users, batch_items, batch_ratings)
                        self.student_optimizer[i].zero_grad()
                        loss.backward()
                        self.student_optimizer[i].step()
                
                # 此处需在model中实现对比学习、双层公平调节、以及元学习内循环
                # 1. 对比学习与公平调节
                # cl_loss = self.client_model[i].contrastive_loss(...)
                # fair_loss = self.client_model[i].fairness_regularization(...)
                # (cl_loss + fair_loss).backward()
                # self.student_optimizer[i].step()

                # 2. 元学习内循环（支持集适应 -> 查询集评估 -> 元梯度）
                # 注意：码序列是整数，需要先放宽为概率分布（Gumbel-Softmax），再求导
                # meta_loss = self.client_model[i].meta_adapt(support_data, query_data, steps=self.meta_steps, lr=self.meta_lr)
            
            self.client_model[i] = self.client_model[i].to('cpu')

    # 客户端上传，融合码序列和鲁棒聚合
    def client_upload(self):
        self.client_sids = [] #每次重新收集
        datasize_sum = 0
        aggstate = {k: 0.0 for k in self.server_paramname}
        
        for i in range(self.client_num):
            datasize = self.dataset_list[i].__len__()
            datasize_sum += datasize
            self.client_model[i] = self.client_model[i].to(self.device)
            
            # 新增：收集码序列，并应用H_u隐私级数截断
            local_sid = self.client_model[i].sid.detach().cpu().numpy()
            H_u = self.H_u_list[i]
            truncated_sid = local_sid[:, :H_u] 
            self.client_sids.append(truncated_sid)
            
            # 原有聚合逻辑，在此处可以将加权平均替换为Trimmed-Mean
            for k in self.server_paramname:
                aggstate[k] = aggstate[k] + self.client_model[i].state_dict()[k] * datasize
            self.client_model[i] = self.client_model[i].to("cpu")

        for k in self.server_paramname:
            aggstate[k] = aggstate[k] / datasize_sum
            # 如果要在此处实现Trimmed-Mean，需在此处先收集所有客户端的张量，排序后去掉极值再平均
        self.server_model.load_state_dict(aggstate, strict=False)

    # 新增码共现图
    def build_code_cooccurrence_graph(self):
        """
        构建码共现图，寻找语义邻居
        对应方案：服务器按码级分层，找到语义邻居。
        """
        self.logger.info("Building Code Co-occurrence Graph...")
        self.semantic_neighbors = []
        
        code_to_users = {}
        for u_idx, sid_seq in enumerate(self.client_sids):
            for h in range(sid_seq.shape[1]): 
                for k in np.unique(sid_seq[:, h]): 
                    if (h, k) not in code_to_users:
                        code_to_users[(h, k)] = []
                    code_to_users[(h, k)].append(u_idx)
                    
        for u_idx in range(self.client_num):
            u_sid_seq = self.client_sids[u_idx]
            neighbor_counter = {}
            
            for h in range(u_sid_seq.shape[1]):
                for k in np.unique(u_sid_seq[:, h]):
                    for other_u in code_to_users.get((h, k), []):
                        if other_u == u_idx: continue
                        if other_u not in neighbor_counter:
                            neighbor_counter[other_u] = 0
                        neighbor_counter[other_u] += 1
            
            neighbors = [uid for uid, count in neighbor_counter.items() if count >= self.sem_threshold]
            self.semantic_neighbors.append(neighbors)
            
        self.logger.info(f"Graph built. Avg semantic neighbors: {np.mean([len(n) for n in self.semantic_neighbors]):.2f}")
    
    # 客户端下载，融合双通道注入
    def client_download(self):
        for i in range(self.client_num):
            private_state = {k: self.server_model.state_dict()[k] for k in self.download_paramname}
            self.client_model[i].load_state_dict(private_state, strict=False)
            self.client_model[i] = self.client_model[i].to(self.device)

            # 新增加，获取双通道邻居并注入本地图
            sem_neigh = self.semantic_neighbors[i] if i < len(self.semantic_neighbors) else []
            beh_neigh = self.behavior_neighbors[i] if i < len(self.behavior_neighbors) else []
            
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

    # 主循环串联
    def fit(self):
        for epoch_i in range(self.epoch):
            start_time = time.time()
            self.epoch_i = epoch_i

            if epoch_i == 0:
                self.client_init()
                self.logger.info(f"Finish init client model in {(time.time() - start_time):.2f}s")

            # 1. 客户端本地训练，包含元学习、CL、公平调节
            self.train_client_model()
            self.logger.info(f"Finish client train in {(time.time() - start_time):.2f}s")

            # 2. 客户端上传，包含H_u隐私截断，码序列上传、鲁棒聚合
            self.client_upload()
            self.logger.info(f"Finish upload in {(time.time() - start_time):.2f}s")

            # 3. 服务器动态更新全局码本
            if epoch_i > 0:
                self.update_data()

            # 4. 新增加，服务端构建码共现图，寻找语义邻居
            self.build_code_cooccurrence_graph()
            self.logger.info(f"Finish Graph construction in {(time.time() - start_time):.2f}s")

            # 5. 客户端下载，全局码本和行为/语义双通道邻居注入
            self.client_download()
            self.logger.info(f"Finish download in {(time.time() - start_time):.2f}s")

            # 6. 验证集评估
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
