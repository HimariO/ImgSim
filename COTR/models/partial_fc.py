import os
import math

import torch
from torch._C import device
import torch.distributed as dist
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.functional import normalize, linear
from torch.nn.parameter import Parameter
from loguru import logger


class ArcFace(nn.Module):
    
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine


class DistPartialFC(Module):
    """
    Author: {Xiang An, Yang Xiao, XuHan Zhu} in DeepGlint,
    Partial FC: Training 10 Million Identities on a Single Machine
    See the original paper:
    https://arxiv.org/abs/2010.05222
    """

    @torch.no_grad()
    def __init__(self, rank, local_rank, world_size, batch_size, resume,
                 margin_softmax, num_classes, sample_rate=1.0, embedding_size=512, prefix="./"):
        """
        rank: int
            Unique process(GPU) ID from 0 to world_size - 1.
        local_rank: int
            Unique process(GPU) ID within the server from 0 to 7.
        world_size: int
            Number of GPU.
        batch_size: int
            Batch size on current rank(GPU).
        resume: bool
            Select whether to restore the weight of softmax.
        margin_softmax: callable
            A function of margin softmax, eg: cosface, arcface.
        num_classes: int
            The number of class center storage in current rank(CPU/GPU), usually is total_classes // world_size,
            required.
        sample_rate: float
            The partial fc sampling rate, when the number of classes increases to more than 2 millions, Sampling
            can greatly speed up training, and reduce a lot of GPU memory, default is 1.0.
        embedding_size: int
            The feature dimension, default is 512.
        prefix: str
            Path for save checkpoint, default is './'.
        """
        super(DistPartialFC, self).__init__()
        #
        self.num_classes: int = num_classes
        self.rank: int = rank
        self.local_rank: int = local_rank
        self.device: torch.device = torch.device("cuda:{}".format(self.local_rank))
        self.world_size: int = world_size
        self.batch_size: int = batch_size
        self.margin_softmax: callable = margin_softmax
        self.sample_rate: float = sample_rate
        self.embedding_size: int = embedding_size
        self.prefix: str = prefix
        self.num_local: int = num_classes // world_size + int(rank < num_classes % world_size)
        self.class_start: int = num_classes // world_size * rank + min(rank, num_classes % world_size)
        self.num_sample: int = int(self.sample_rate * self.num_local)

        self.weight_name = os.path.join(self.prefix, "rank_{}_softmax_weight.pt".format(self.rank))
        self.weight_mom_name = os.path.join(self.prefix, "rank_{}_softmax_weight_mom.pt".format(self.rank))

        if resume:
            try:
                self.weight: torch.Tensor = torch.load(self.weight_name)
                self.weight_mom: torch.Tensor = torch.load(self.weight_mom_name)
                if self.weight.shape[0] != self.num_local or self.weight_mom.shape[0] != self.num_local:
                    raise IndexError
                logger.info("softmax weight resume successfully!")
                logger.info("softmax weight mom resume successfully!")
            except (FileNotFoundError, KeyError, IndexError):
                self.weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device)
                self.weight_mom: torch.Tensor = torch.zeros_like(self.weight)
                logger.info("softmax weight init!")
                logger.info("softmax weight mom init!")
        else:
            self.weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device)
            self.weight_mom: torch.Tensor = torch.zeros_like(self.weight)
            logger.info("softmax weight init successfully!")
            logger.info("softmax weight mom init successfully!")
        self.stream: torch.cuda.Stream = torch.cuda.Stream(local_rank)

        self.index = None
        if int(self.sample_rate) == 1:
            self.update = lambda: 0
            self.sub_weight = Parameter(self.weight)
            self.sub_weight_mom = self.weight_mom
        else:
            self.sub_weight = Parameter(torch.empty((0, 0)).cuda(local_rank))

    def save_params(self):
        """ Save softmax weight for each rank on prefix
        """
        torch.save(self.weight.data, self.weight_name)
        torch.save(self.weight_mom, self.weight_mom_name)

    @torch.no_grad()
    def sample(self, total_label):
        """
        Sample all positive class centers in each rank, and random select neg class centers to filling a fixed
        `num_sample`.

        total_label: tensor
            Label after all gather, which cross all GPUs.
        """
        index_positive = (self.class_start <= total_label) & (total_label < self.class_start + self.num_local)
        total_label[~index_positive] = -1
        total_label[index_positive] -= self.class_start
        if int(self.sample_rate) != 1:
            positive = torch.unique(total_label[index_positive], sorted=True)
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local], device=self.device)
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1]
                index = index.sort()[0]
            else:
                index = positive
            self.index = index
            total_label[index_positive] = torch.searchsorted(index, total_label[index_positive])
            self.sub_weight = Parameter(self.weight[index])
            self.sub_weight_mom = self.weight_mom[index]

    def forward(self, total_features, norm_weight):
        """ Partial fc forward, `logits = X * sample(W)`
        """
        torch.cuda.current_stream().wait_stream(self.stream)
        logits = linear(total_features, norm_weight)
        return logits

    @torch.no_grad()
    def update(self):
        """ Set updated weight and weight_mom to memory bank.
        """
        self.weight_mom[self.index] = self.sub_weight_mom
        self.weight[self.index] = self.sub_weight

    def prepare(self, label, optimizer):
        """
        get sampled class centers for cal softmax.

        label: tensor
            Label tensor on each rank.
        optimizer: opt
            Optimizer for partial fc, which need to get weight mom.
        """
        with torch.cuda.stream(self.stream):
            total_label = torch.zeros(
                size=[self.batch_size * self.world_size], device=self.device, dtype=torch.long)
            dist.all_gather(list(total_label.chunk(self.world_size, dim=0)), label)
            self.sample(total_label)
            
            optimizer.state.pop(optimizer.param_groups[-1]['params'][0], None)
            optimizer.param_groups[-1]['params'][0] = self.sub_weight
            optimizer.state[self.sub_weight]['momentum_buffer'] = self.sub_weight_mom
            norm_weight = normalize(self.sub_weight)
            return total_label, norm_weight

    def forward_backward(self, label, features, optimizer):
        """
        Partial fc forward and backward with model parallel

        label: tensor
            Label tensor on each rank(GPU)
        features: tensor
            Features tensor on each rank(GPU)
        optimizer: optimizer
            Optimizer for partial fc

        Returns:
        --------
        x_grad: tensor
            The gradient of features.
        loss_v: tensor
            Loss value for cross entropy.
        """
        total_label, norm_weight = self.prepare(label, optimizer)
        total_features = torch.zeros(
            size=[self.batch_size * self.world_size, self.embedding_size], device=self.device)
        dist.all_gather(list(total_features.chunk(self.world_size, dim=0)), features.data)
        total_features.requires_grad = True

        logits = self.forward(total_features, norm_weight)
        logits = self.margin_softmax(logits, total_label)

        with torch.no_grad():
            max_fc = torch.max(logits, dim=1, keepdim=True)[0]
            dist.all_reduce(max_fc, dist.ReduceOp.MAX)

            # calculate exp(logits) and all-reduce
            logits_exp = torch.exp(logits - max_fc)
            logits_sum_exp = logits_exp.sum(dim=1, keepdims=True)
            dist.all_reduce(logits_sum_exp, dist.ReduceOp.SUM)

            # calculate prob
            logits_exp.div_(logits_sum_exp)

            # get one-hot
            grad = logits_exp
            index = torch.where(total_label != -1)[0]
            one_hot = torch.zeros(size=[index.size()[0], grad.size()[1]], device=grad.device)
            one_hot.scatter_(1, total_label[index, None], 1)

            # calculate loss
            loss = torch.zeros(grad.size()[0], 1, device=grad.device)
            loss[index] = grad[index].gather(1, total_label[index, None])
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            loss_v = loss.clamp_min_(1e-30).log_().mean() * (-1)

            # calculate grad
            grad[index] -= one_hot
            grad.div_(self.batch_size * self.world_size)

        logits.backward(grad)
        if total_features.grad is not None:
            total_features.grad.detach_()
        x_grad: torch.Tensor = torch.zeros_like(features, requires_grad=True)
        # feature gradient all-reduce
        dist.reduce_scatter(x_grad, list(total_features.grad.chunk(self.world_size, dim=0)))
        x_grad = x_grad * self.world_size
        # backward backbone
        return x_grad, loss_v


class PartialFC(Module):
    
    def __init__(self, num_classes: int, embed_dim: int, sample_ratio=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.sample_ratio = sample_ratio
        self.weights = nn.Embedding(num_classes, embed_dim, sparse=True)
    
    def sample(self, positive: torch.Tensor):
        # postive_classes (b, 1) or (b,)
        n = len(positive)
        num_sample = max(math.ceil(self.sample_ratio * self.num_classes) - n, n * 2)
        # positive = torch.unique(total_label[index_positive], sorted=True)
        perm = torch.rand(size=[self.num_classes], device=positive.device)
        perm[positive] = -1
        negative = torch.topk(perm, k=num_sample, sorted=True)[1]
        # index = index.sort()[0]
        return negative

    def forward(self, x: torch.Tensor, y: torch.LongTensor):
        assert y.dtype in [torch.int32, torch.int64]
        positive = torch.unique(y, sorted=True)
        pos_idx = torch.arange(len(positive), device=y.device)
        # val_idx = torch.stack([positive, pos_idx], dim=1)
        idx_map = torch.zeros([self.num_classes], dtype=torch.int64, device=y.device) - 1
        idx_map[positive] = pos_idx
        remap_y = idx_map[y]
        
        pos_w_mtx = self.weights(positive)
        neg_class = self.sample(positive)
        neg_w_mtx = self.weights(neg_class)
        w_mtx = torch.cat([pos_w_mtx, neg_w_mtx], dim=0)
        logits = linear(normalize(x), normalize(w_mtx))
        # breakpoint()
        return logits, remap_y



class ArcMarginProduct(Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, fc_layer: Module, s=30.0, m=0.10, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m = m
        self.fc = fc_layer
        # self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = Parameter(torch.tensor(math.cos(m)), requires_grad=False)
        self.sin_m = Parameter(torch.tensor(math.sin(m)), requires_grad=False)
        self.th = Parameter(torch.tensor(math.cos(math.pi - m)), requires_grad=False)
        self.mm = Parameter(torch.tensor(math.sin(math.pi - m) * m), requires_grad=False)

    def forward(self, input: torch.Tensor, label: torch.LongTensor):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine, label = self.fc(input, label)
        cosine = cosine.float()

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-7, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output, label


def test_fc():
    fc = PartialFC(10000, 256, sample_ratio=0.1)
    x = torch.normal(
        mean=torch.zeros([8, 256]),
        std=torch.ones([8, 256])
    )
    y = torch.tensor([66, 77, 88, 100, 20, 30, 100, 55], dtype=torch.int64)
    logits, new_y = fc.forward(x, y)
    print(logits.shape)
    print(new_y)


def test_arc_fc():
    x = torch.normal(
        mean=torch.zeros([8, 256]),
        std=torch.ones([8, 256])
    )
    y = torch.tensor([66, 77, 88, 100, 20, 30, 100, 55], dtype=torch.int64)
    
    fc = PartialFC(10000, 256, sample_ratio=0.1)
    arc = ArcMarginProduct(fc, )
    arc_logit, label = arc(x, y)
    print(arc_logit)
    print(arc_logit.shape)
    print(label)


if __name__ == '__main__':
    test_arc_fc()