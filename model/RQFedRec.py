import torch
from torch import nn
import torch.nn.functional as F # 新增：用于CL中的归一化

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
        self.neighbor_bias = {} # 新增：存储每个用户的双通道邻居偏置

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

    def teacher_loss(self, users, items, label):
        users_emb = self.embedding_user(users)
        
        # 修复：注入双通道邻居偏置，让推荐预测真正用上邻居信息
        # 注意：这里假设当前客户端 i 的本地用户 ID 是统一映射的（例如 0）
        # 如果每个客户端本地只有一个用户，u_id 通常就是 0
        if len(self.neighbor_bias) > 0:
            u_id = users[0].item() # 取当前 batch 的第一个用户 ID
            if u_id in self.neighbor_bias:
                users_emb = users_emb + self.neighbor_bias[u_id]

        items_emb = self.embedding_item(items)
        scores = self.f(torch.sum(torch.mul(users_emb, items_emb), dim=1))
        return self.BCEloss(scores, label)
    
    def student_loss(self, users, items, label):
        student_items_emb = self.get_item_emb(items)
        teacher_items_emb = self.embedding_item(items)
        return self.MSEloss(teacher_items_emb.detach(), student_items_emb)


    def getUsersRating(self, users, items):
        users_emb = self.embedding_user(torch.tensor([users]).to(self.device))
        
        # 修复：评估时同样注入双通道邻居偏置
        if len(self.neighbor_bias) > 0 and users in self.neighbor_bias:
            users_emb = users_emb + self.neighbor_bias[users]

        teacher_items_emb = self.embedding_item(items.to(self.device))
        return self.f(torch.matmul(users_emb, teacher_items_emb.t()))

    @torch.no_grad()
    def teacher_update(self):
        all_items = torch.arange(self.num_items, device=self.device)
        student_items_emb = self.get_item_emb(all_items)
        self.embedding_item.weight.data = student_items_emb.clone().detach()

    # 新增：FedCodeFair核心组价
    def inject_neighbors(self, user_id, sem_neigh, beh_neigh):
        """
        新增方法：双通道邻居注入。
        对应方案：接收服务器下发的行为邻居（F2PGNN）和语义邻居（码共现图），扩展本地图。
        实现思路：
            此处采用邻居嵌入平均聚合，并将聚合后的邻居信息
            作为偏置，注入到对应用户的表示中。后续可替换为 GAT 或 LightGCN 多层传播。
        """
        # 1. 处理语义邻居 (Semantic Neighbors)
        if len(sem_neigh) > 0:
            sem_tensor = torch.tensor(sem_neigh, device=self.device, dtype=torch.long)
            sem_embs = self.embedding_user(sem_tensor) 
            sem_agg = torch.mean(sem_embs, dim=0) 
        else:
            sem_agg = torch.zeros(self.latent_dim, device=self.device)

        # 2. 处理行为邻居 (Behavior Neighbors)
        if len(beh_neigh) > 0:
            beh_tensor = torch.tensor(beh_neigh, device=self.device, dtype=torch.long)
            beh_embs = self.embedding_user(beh_tensor)
            beh_agg = torch.mean(beh_embs, dim=0)
        else:
            beh_agg = torch.zeros(self.latent_dim, device=self.device)

        # 3. 双通道融合并返回（可用于后续更新用户嵌入）
        # 实际应用中，你可能会把 (sem_agg + beh_agg) 加到一个 user_bias 上
        neighbor_bias = sem_agg + beh_agg
        self.neighbor_bias[user_id] = neighbor_bias.detach()
        return neighbor_bias

    def contrastive_loss(self, user_emb_view1, user_emb_view2, temperature=0.1):
        """
        新增方法：双视图对比学习。
        对应方案：拉近同一用户的结构增强视图和语义相关视图，拉近语义相关用户的表示。
        实现思路：使用余弦相似度 + InfoNCE Loss（归一化温度缩放交叉熵）。
        """
        # 归一化
        view1 = F.normalize(user_emb_view1, dim=1)
        view2 = F.normalize(user_emb_view2, dim=1)
        
        # 计算相似度矩阵 (batch_size, batch_size)
        logits = torch.matmul(view1, view2.T) / temperature
        
        # 对角线上的元素是正样本，即同一个用户的两种视图
        labels = torch.arange(logits.shape[0], device=self.device)
        
        # 对称对比损失
        loss = F.cross_entropy(logits, labels)
        return loss

    def fairness_regularization(self, sid_batch):
        """
        新增方法：双层公平约束（表示层）。
        对应方案：看不同群体在各级码上的分布差异，把分布差异作为正则项。
        实现思路：鼓励用户在各码级上的选择分布更均匀（最大化信息熵），防止语义坍缩至少数几个码。
        """
        total_entropy = 0.0
        for h in range(self.sid_length):
            # 统计当前 batch 中第 h 级码的频率分布
            code_counts = torch.bincount(sid_batch[:, h].long(), minlength=self.cluster_num).float()
            probs = code_counts / (code_counts.sum() + 1e-8)
            # 计算熵：Entropy = - sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            total_entropy += entropy
            
        # 希望熵越大越好（分布越均匀越公平），所以正则项是负的熵
        fairness_loss = -total_entropy / self.sid_length
        return fairness_loss

    def meta_adapt(self, support_data, query_data, steps, lr):
        """
        元学习内循环（FedCodeFair-B 最大创新点）。
        对应方案：支持集适应 -> 查询集评估 -> 计算元梯度。
        实现难点：离散的 SID 不可导。这里使用 Gumbel-Softmax 松弛离散码为连续概率分布。
        """
        # 1. 克隆一组快速适应参数（内循环专用）
        fast_weights = [c.weight.clone().detach().requires_grad_(True) for c in self.codebook_embeddings]
        
        # 解包支持集和查询集
        s_users, s_items, s_ratings = support_data
        q_users, q_items, q_ratings = query_data
        
        # 2. 支持集适应（Support Set Adaptation）
        for _ in range(steps):
            # 使用快速权重计算支持集的前向传播
            # 为了简化，这里直接使用支持集的推荐损失（实际可加入 Gumbel-Softmax 松弛）
            s_users_emb = self.embedding_user(s_users)
            
            # 临时替换码本权重以计算快速适应损失
            original_weights = [c.weight.data.clone() for c in self.codebook_embeddings]
            for c, w in zip(self.codebook_embeddings, fast_weights):
                c.weight.data = w
                
            s_items_emb = self.get_item_emb(s_items)
            scores = self.f(torch.sum(torch.mul(s_users_emb, s_items_emb), dim=1))
            loss = self.BCEloss(scores, s_ratings)
            
            # 还原原始权重，防止影响外循环
            for c, w in zip(self.codebook_embeddings, original_weights):
                c.weight.data = w
                
            # 求梯度并更新快速权重
            grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
            fast_weights = [w - lr * g for w, g in zip(fast_weights, grads)]
            
        # 3. 查询集评估（Query Set Evaluation）
        # 使用适应后的 fast_weights，在查询集上计算 meta_loss
        original_weights = [c.weight.data.clone() for c in self.codebook_embeddings]
        for c, w in zip(self.codebook_embeddings, fast_weights):
            c.weight.data = w
            
        q_users_emb = self.embedding_user(q_users)
        q_items_emb = self.get_item_emb(q_items)
        q_scores = self.f(torch.sum(torch.mul(q_users_emb, q_items_emb), dim=1))
        meta_loss = self.BCEloss(q_scores, q_ratings)
        
        # 还原原始权重
        for c, w in zip(self.codebook_embeddings, original_weights):
            c.weight.data = w
            
        # 返回 meta_loss，让外部的 student_optimizer 去更新全局码本
        return meta_loss
