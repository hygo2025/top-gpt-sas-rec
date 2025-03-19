import numpy as np


def evaluate(model, dataset, sequence_length, isvalid=False):
    [train, valid, test, usernum, itemnum] = dataset

    NDCG = 0.0
    HT = 0.0
    evaluate_usernum = 0.0
    users = range(1, usernum + 1)

    for u in users:
        # 判断是验证集还是测试集,如果训练集没有数据则跳过，如果验证集或者测试集没有数据则跳过
        if isvalid and (len(train[u]) < 1 or len(valid[u]) < 1):
            continue
        elif not isvalid and (len(train[u]) < 1 or len(test[u]) < 1):
            continue

        # 初始化序列 根据原论文 如果真实交互的序列未达到指定步长，要使用填充项，即0进行填充 例如seq = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,item]
        seq = np.zeros([sequence_length], dtype=np.int32)
        idx = sequence_length - 1 #取出最后一个位置

        # 判断是验证阶段还是测试阶段,如果是测试阶段则取出验证集的最后一个物品作为真实交互物品
        if not isvalid:
            seq[idx] = valid[u][0]
            idx -= 1

        # 取出训练集中的物品交互序列
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        # 取出训练集中的物品交互序列
        rated = set(train[u])
        # 将物品加入到候选物品集中
        if isvalid:
            item_idx = [valid[u][0]]
        else:
            item_idx = [test[u][0]]

        # 在这里模仿pmixer的代码根据100个数量 选择候选物品集
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        # 预测，为什么要在seq外层再套个数组？ 为了适应模型的输入，模型的输入第一维是batch_size
        # 为什么要套个np.array()? 因为np.array转为Tensor时速度更快
        #为什么要加个负号？ 其实加不加都行，因为后续使用的argsort()默认是升序，加个负号能够实现意义上的降序
        predictions = - model.predict(np.array([seq]), np.array(item_idx))
        #为什么要取[0]，因为模型的输入第一维是batch_size，所以模型的输出也是batch_size，这里batch_size=1，所以取[0]
        predictions = predictions[0]
        #这个是参考原论文的代码，这行代码当时我在想为啥要这样，后来发现是为了取出预测的物品的索引，读者可以认真思考一下，
        #经过两轮argsort()后，predictions中的元素就是对应原，所以取出第一个索引就是预测的物品的排名
        rank = predictions.argsort().argsort()[0].item()

        # 如果物品在前10名内，则命中，计算其NDCG和HR
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        evaluate_usernum += 1
    # 返回NDCG和HR的平均值
    return NDCG / evaluate_usernum, HT / evaluate_usernum