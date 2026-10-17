import torch
from torch import nn
import torch.nn.functional as F

from .base import BasicModel


class RQFedRec(BasicModel):
    def __init__(self, config, dataset):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = config
        self.dataset = dataset
        self.__init_weight(config)

    def __init_weight(self, config):
        self.num_users = self.dataset.user_num
        self.num_items = self.dataset.item_num
        self.latent_dim = config.getint('latent_dim')
        self.weight_decay = config.getfloat('weight_decay', fallback=0.0)
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.BCEloss = nn.BCELoss()
        self.MSEloss = nn.MSELoss()
        self.neighbor_bias = {} # 【修复】存储双通道邻居偏置

    def init_codebook(self, sid, cluster_num):
        self.sid_length = sid.shape[1]
        self.register_buffer('sid', sid.clone().long())
        self.sid = self.sid.to(self.device)
        self.cluster_num = cluster_num

        self.codebook_embeddings = torch.nn.ModuleList()
        for _ in range(self.sid_length):
            codebook = torch.nn.Embedding(num_embeddings=self.cluster_num, embedding_dim=self.latent_dim)
            nn.init.normal_(codebook.weight, std=0.1)
            self.codebook_embeddings.append(codebook)

    def get_item_emb(self, item):
        item_emb = torch.sum(torch.stack([self.codebook_embeddings[_](self.sid[item, _]) for _ in range(self.sid_length)]), dim=0)
        return item_emb

    # 【修复】在教师损失中加入双通道邻居偏置
    def teacher_loss(self, users, items, label):
        users_emb = self.embedding_user(users)
        
        if len(self.neighbor_bias) > 0:
            u_id = users[0].item()
            if u_id in self.neighbor_bias:
                users_emb = users_emb + self.neighbor_bias[u_id]

        items_emb = self.embedding_item(items)
        scores = self.f(torch.sum(torch.mul(users_emb, items_emb), dim=1))
        return self.BCEloss(scores, label)

    def student_loss(self, users, items, label):
        student_items_emb = self.get_item_emb(items)
        teacher_items_emb = self.embedding_item(items)
        return self.MSEloss(teacher_items_emb.detach(), student_items_emb)

    # 【修复】在评估预测中加入双通道邻居偏置
    def getUsersRating(self, users, items):
        users_emb = self.embedding_user(torch.tensor([users]).to(self.device))
        
        if len(self.neighbor_bias) > 0 and users in self.neighbor_bias:
            users_emb = users_emb + self.neighbor_bias[users]

        teacher_items_emb = self.embedding_item(items.to(self.device))
        return self.f(torch.matmul(users_emb, teacher_items_emb.t()))

    @torch.no_grad()
    def teacher_update(self):
        all_items = torch.arange(self.num_items, device=self.device)
        student_items_emb = self.get_item_emb(all_items)
        self.embedding_item.weight.data = student_items_emb.clone().detach()

    # ==================== FedCodeFair 核心组件 ====================
    def inject_neighbors(self, user_id, sem_neigh, beh_neigh):
        """
        新增方法：双通道邻居注入。
        """
        # 【修复1：安全过滤】过滤掉超出用户嵌入表范围的邻居索引，防止 CUDA 越界报错
        sem_neigh = [u for u in sem_neigh if u < self.num_users]
        beh_neigh = [u for u in beh_neigh if u < self.num_users]

        if len(sem_neigh) > 0:
            sem_tensor = torch.tensor(sem_neigh, device=self.device, dtype=torch.long)
            sem_embs = self.embedding_user(sem_tensor) 
            sem_agg = torch.mean(sem_embs, dim=0) 
        else:
            sem_agg = torch.zeros(self.latent_dim, device=self.device)

        if len(beh_neigh) > 0:
            beh_tensor = torch.tensor(beh_neigh, device=self.device, dtype=torch.long)
            beh_embs = self.embedding_user(beh_tensor)
            beh_agg = torch.mean(beh_embs, dim=0)
        else:
            beh_agg = torch.zeros(self.latent_dim, device=self.device)

        neighbor_bias = sem_agg + beh_agg
        self.neighbor_bias[user_id] = neighbor_bias.detach()
        return neighbor_bias

    def contrastive_loss(self, user_emb_view1, user_emb_view2, temperature=0.1):
        """
        双视图对比学习。
        """
        view1 = F.normalize(user_emb_view1, dim=1)
        view2 = F.normalize(user_emb_view2, dim=1)
        logits = torch.matmul(view1, view2.T) / temperature
        labels = torch.arange(logits.shape[0], device=self.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def fairness_regularization(self, sid_batch):
        """
        双层公平约束（表示层）。
        """
        total_entropy = 0.0
        for h in range(self.sid_length):
            code_counts = torch.bincount(sid_batch[:, h].long(), minlength=self.cluster_num).float()
            probs = code_counts / (code_counts.sum() + 1e-8)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            total_entropy += entropy
        fairness_loss = -total_entropy / self.sid_length
        return fairness_loss

    # 【修复2：计算图断裂】用 F.embedding 代替 .data 赋值
    def meta_adapt(self, support_data, query_data, steps, lr):
        """
        元学习内循环。
        """
        fast_weights = [c.weight.clone().detach().requires_grad_(True) for c in self.codebook_embeddings]
        
        s_users, s_items, s_ratings = support_data
        q_users, q_items, q_ratings = query_data
        
        # 辅助函数：使用指定的权重计算物品嵌入，保持计算图连通
        def get_emb_with_weights(items, weights):
            item_emb = torch.sum(torch.stack([F.embedding(self.sid[items, _], weights[_]) for _ in range(self.sid_length)]), dim=0)
            return item_emb

        for _ in range(steps):
            s_users_emb = self.embedding_user(s_users)
            s_items_emb = get_emb_with_weights(s_items, fast_weights)
            scores = self.f(torch.sum(torch.mul(s_users_emb, s_items_emb), dim=1))
            loss = self.BCEloss(scores, s_ratings)
            
            grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
            fast_weights = [w - lr * g for w, g in zip(fast_weights, grads)]
            
        q_users_emb = self.embedding_user(q_users)
        q_items_emb = get_emb_with_weights(q_items, fast_weights)
        q_scores = self.f(torch.sum(torch.mul(q_users_emb, q_items_emb), dim=1))
        meta_loss = self.BCEloss(q_scores, q_ratings)
        
        return meta_loss
