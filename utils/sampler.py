import torch



import utils.tensor


class Sampler:
    def __init__(self, probabity):
        self.probabity = torch.abs(probabity)
        assert self.probabity.shape[0] == 1
        assert self.probabity.shape[1] == 1


        self.height =  self.probabity.shape[2]
        self.width =  self.probabity.shape[3]

        cdf = torch.cumsum(self.probabity.reshape(-1))
        self.cdf = cdf/cdf[-1]


    def sample(self, s):
        batch_size = s.shape[0]

        tt = utils.tensor.Type(s)

        idx = dd
        pos_y = idx/ self.width
        pos_x = idx % self.width

        pos_y = pos_y + torch.rand([batch_size], **tt.same_type())/self.height
        pos_x = pos_x +  torch.rand([batch_size], **tt.same_type())/self.width

        return torch.cat([pos_y, pos_x], dim=1)
