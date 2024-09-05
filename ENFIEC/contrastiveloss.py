import torch
from torch import nn
import torch.nn.functional as F

from ENFIEC import config


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, emb_i, emb_j, emb_k):
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        z_k = F.normalize(emb_k, dim=1)
        bach_size_i = z_i.shape[0]  #8
        bach_size_j = z_j.shape[0]
        bach_size_k = z_k.shape[0]
        representations_row = z_i
        representations_col = torch.cat([z_j, z_k], dim=0)
        similarity_matrix = F.cosine_similarity(representations_row.unsqueeze(1), representations_col.unsqueeze(0),
                                                dim=2)
        print("3e与4a的平均余弦相似度为：", torch.mean(similarity_matrix[:, :bach_size_j]))
        print("3e与3l的平均余弦相似度为：", torch.mean(similarity_matrix[:, bach_size_i:]))

        def l_ij(k, h):
            sim_k_h = similarity_matrix[k, h]
            numerator = torch.exp(sim_k_h / self.temperature)
            one_for_not_k_h = torch.zeros((bach_size_j + bach_size_k,)).to(emb_i.device).scatter_(0, torch.tensor(
                [i for i in range(bach_size_j + bach_size_k) if i == k or i >= bach_size_i]).to(emb_i.device), 1.0).to(
                emb_i.device)
            denominator = torch.sum(one_for_not_k_h * torch.exp(similarity_matrix[k, :] / self.temperature))
            loss_ij = -torch.log(numerator / denominator)
            return loss_ij.squeeze(0)

        loss = 0.0
        for k in range(0, bach_size_i):
            for h in range(0, bach_size_j):
                if k == h:
                    loss += l_ij(k, h)

        return (1.0 / bach_size_i) * loss


class ContrastiveLoss2Tradition(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss2Tradition, self).__init__()
        self.temperature = temperature

    def forward(self, emb_i, emb_j, emb_k):
        z_i = F.normalize(emb_i, dim=1)  # 3e
        z_j = F.normalize(emb_j, dim=1)  # 4a
        z_k = F.normalize(emb_k, dim=1)  # 3l
        bach_size_i = z_i.shape[0]  # 8
        bach_size_j = z_j.shape[0]
        bach_size_k = z_k.shape[0]
        representations_col = torch.cat([z_i, z_j, z_k], dim=0)  # 8.17
        similarity_matrix = F.cosine_similarity(representations_col.unsqueeze(1), representations_col.unsqueeze(0),
                                                dim=2)

        def l_ij(three_e, four_a):
            sim_3e_4a = similarity_matrix[three_e, four_a]
            numerator = torch.exp(sim_3e_4a / self.temperature)
            one_for_not_3e_4a = (
                torch.zeros((bach_size_i + bach_size_j + bach_size_k,)).to(emb_i.device).scatter_(0, torch.tensor(
                    [i for i in range(bach_size_i + bach_size_j + bach_size_k) if i != three_e]).to(emb_i.device),
                                                                                                  1.0).to(
                    emb_i.device))

            denominator = torch.sum(one_for_not_3e_4a * torch.exp(similarity_matrix[three_e, :] / self.temperature))

            return numerator, denominator

        loss_i = 0.0
        size = bach_size_i + bach_size_j + bach_size_k
        for i in range(0, bach_size_i + bach_size_j):
            # loss_j = 0.0
            numerator_tensor = torch.tensor(0.0).to(config.DEVICE)
            for j in range(0, size - bach_size_k):
                if i != j:
                    numerator, denominator = l_ij(i, j)
                else:
                    numerator = torch.tensor(0.0).to(config.DEVICE)
                numerator_tensor += numerator
            numerator_tensor /= (bach_size_i + bach_size_j - 1)  # 8.18
            loss_i += -torch.log(numerator_tensor / denominator)

        for i in range(bach_size_i + bach_size_j, size):
            numerator_tensor = torch.tensor(0.0).to(config.DEVICE)
            for j in range(size - bach_size_k, size):
                if i != j:
                    numerator, denominator = l_ij(i, j)
                else:
                    numerator = torch.tensor(0.0).to(config.DEVICE)
                numerator_tensor += numerator
            numerator_tensor /= (bach_size_k - 1)  # 8.19
            loss_i += -torch.log(numerator_tensor / denominator)

        return (1.0 / size) * loss_i


if __name__ == "__main__":
    I = torch.rand((3, 3))
    J = torch.rand((4, 3))
    loss = ContrastiveLoss(0.5)
    lo = loss(I, J)
    print(lo)
