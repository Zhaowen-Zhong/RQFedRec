import torch
from torch import nn

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
        items_emb = self.embedding_item(items)
        scores = self.f(torch.sum(torch.mul(users_emb, items_emb), dim=1))
        return self.BCEloss(scores, label)

    def student_loss(self, users, items, label):
        student_items_emb = self.get_item_emb(items)
        teacher_items_emb = self.embedding_item(items)
        return self.MSEloss(teacher_items_emb.detach(), student_items_emb)

    def getUsersRating(self, users, items):
        users_emb = self.embedding_user(torch.tensor([users]).to(self.device))
        teacher_items_emb = self.embedding_item(items.to(self.device))
        return self.f(torch.matmul(users_emb, teacher_items_emb.t()))

    @torch.no_grad()
    def teacher_update(self):
        all_items = torch.arange(self.num_items, device=self.device)
        student_items_emb = self.get_item_emb(all_items)
        self.embedding_item.weight.data = student_items_emb.clone().detach()
