from torch import nn, rsub
import torch.nn.functional as F
# from transformers import AutoModel
import torch



class Teacher(torch.nn.Module):
    def __init__(self, idx2asp, asp_cnt, general_asp) -> None:
        super(Teacher, self).__init__()
        self.idx2asp = idx2asp
        self.asp_cnt = asp_cnt
        self.general_asp = general_asp
        
    def forward(self, bow, zs):
        """Teacher
        Args:
            bow (torch.tensor): [B, bow_size]
            zs  (torch.tensor): [num_asp, bow_size]
        Returns:
            : [B, asp_cnt]
        """
        # for each aspect

        result = []

        for i in range(self.asp_cnt):
            step1 = (bow[:,i,:] * zs[i,:].unsqueeze(0))
            s2 = step1.sum(-1)
            s2 = s2.unsqueeze(-1)
            result.append(s2)

        result = torch.cat(result, -1)
        mask = bow.sum(-1).sum(-1) == 0
        result[mask, self.general_asp] = 1
        result = torch.softmax(result, -1)

        return result

