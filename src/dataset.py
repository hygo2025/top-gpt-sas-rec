import numpy as np
from torch.utils.data import Dataset


class MovielensDataset(Dataset):
    def __init__(self, train_data, sequence_length, usernum, itemnum):
        self.train_data = train_data
        self.sequence_length = sequence_length
        self.usernum = usernum
        self.itemnum = itemnum

    def __len__(self):
        return len(self.train_data)

    def random_neg(self, l, r, ts):
        t = np.random.randint(l, r)
        while t in ts:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, index):
        # 注意这里的getitem是用于dataloader读取数据的，所以这里的index是dataloader传入的，但是这里的index是无用的，因为我们是随机取的数据
        userid = np.random.randint(1, self.usernum + 1)
        while len(self.train_data[userid]) <= 1: userid = np.random.randint(1, self.usernum + 1)

        seq = np.zeros([self.sequence_length], dtype=np.int32)
        pos = np.zeros([self.sequence_length], dtype=np.int32)
        neg = np.zeros([self.sequence_length], dtype=np.int32)

        # 根据论文中的描述，我们需要将序列中的物品分为正样本和负样本，正样本就是序列中的物品，负样本就是随机抽取的物品
        # 而我们的序列 你可以发现 seq和pos是不一样的 seq是序列中的物品，而pos是seq序列向后移动一位的结果，这样就可以得到正样本
        # 也就是说 pos是真正训练集序列的后50个物品，而seq要整体向左移动一位 因为要以seq预测pos
        # 根据原论文  假设我们取的序列长度是50，而真实序列长度不够50  则记得再前面补0 即填充项，然后在后续的处理中过滤掉填充项
        offset_item = self.train_data[userid][-1]
        idx = self.sequence_length - 1
        ts = set(self.train_data[userid])
        for i in reversed(self.train_data[userid][:-1]):
            seq[idx] = i
            pos[idx] = offset_item
            if offset_item != 0: neg[idx] = self.random_neg(1, self.itemnum + 1, ts)
            offset_item = i
            idx -= 1
            if idx == -1: break
        return (userid, seq, pos, neg)