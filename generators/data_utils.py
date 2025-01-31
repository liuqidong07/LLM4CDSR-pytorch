# here put the import lib
import copy
import numpy as np


def random_neq(l, r, s=[]):    # 在l-r之间随机采样一个数，这个数不能在列表s中
    
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def truncate_padding(inter, domain_mask, max_len, item_numA, item_numB):

    non_neg = copy.deepcopy(inter)
    seq = np.zeros([max_len], dtype=np.int32)
    pos = np.zeros([max_len], dtype=np.int32)
    neg = np.zeros([max_len], dtype=np.int32)
    mask = np.ones([max_len], dtype=np.int32) * -1

    if len(inter)>0:    # for CDSR, it can be void sequence
        nxt = inter[-1]
        idx = max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            mask_idx = idx - max_len - 1

            if domain_mask[mask_idx] == 0:
                neg[idx] = random_neq(1, item_numA, non_neg)
            elif domain_mask[mask_idx] == 1:
                neg[idx] = random_neq(item_numA+1, item_numA+item_numB, non_neg)
            else:
                raise ValueError
            mask[idx] = domain_mask[mask_idx+1]
            nxt = i
            idx -= 1
            if idx == -1:
                break
            if -mask_idx == len(domain_mask):
                break
        
        true_len = len(seq[seq>0]) + 1
        if true_len > max_len:
            mask_len = 0
            positions = list(range(1, max_len+1))
        else:
            mask_len = max_len - (true_len - 1)
            positions = list(range(1, true_len-1+1))
        
        positions= positions[-max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)
    else:
        positions = [0] * max_len
        positions = np.array(positions)

    return seq, pos, neg, positions, mask
