import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter
from data_loader.feature_utils import load_data2, load_npz
from numpy import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Arguments
args = get_citation_args()


def simi(output):  # new_version
    a = output.norm(dim=1)[:, None]
    the_ones = torch.ones_like(a)
    a = torch.where(a==0, the_ones, a)
    a_norm = output / a
    b_norm = output / a
    res = 5 * (torch.mm(a_norm, b_norm.transpose(0, 1)) + 1)

    return res

# by rows
def idcg_computation(x_sorted_scores, top_k):
    c = 2 * torch.ones_like(x_sorted_scores)[:top_k]
    numerator = c.pow(x_sorted_scores[:top_k]) - 1
    denominator = torch.log2(2 + torch.arange(x_sorted_scores[:top_k].shape[0], dtype=torch.float)).cuda()
    final = numerator / denominator

    return torch.sum(final)

# by rows
def dcg_computation(score_rank, top_k):
    c = 2 * torch.ones_like(score_rank)[:top_k]
    numerator = c.pow(score_rank[:top_k]) - 1
    denominator = torch.log2(2 + torch.arange(score_rank[:top_k].shape[0], dtype=torch.float))
    final = numerator / denominator

    return torch.sum(final)


def ndcg_exchange_abs(x_corresponding, j, k, idcg, top_k):
    new_score_rank = x_corresponding
    dcg1 = dcg_computation(new_score_rank, top_k)
    the_index = np.arange(new_score_rank.shape[0])
    temp = the_index[j]
    the_index[j] = the_index[k]
    the_index[k] = temp
    new_score_rank = new_score_rank[the_index]
    dcg2 = dcg_computation(new_score_rank, top_k)

    return torch.abs((dcg1 - dcg2) / idcg)


def err_computation(score_rank, top_k):
    the_maxs = torch.max(score_rank).repeat(1, score_rank.shape[0])
    c = 2 * torch.ones_like(score_rank)
    score_rank = (( c.pow(score_rank) - 1) / c.pow(the_maxs))[0]
    the_ones = torch.ones_like(score_rank)
    new_score_rank = torch.cat((the_ones, 1 - score_rank))

    for i in range(score_rank.shape[0] - 1):
        score_rank = torch.mul(score_rank, new_score_rank[-score_rank.shape[0] - 1 - i : -1 - i])
    the_range = torch.arange(0., score_rank.shape[0]) + 1

    final = (1 / the_range[0:]) * score_rank[0:]

    return torch.sum(final)



def err_exchange_abs(x_corresponding, j, k, top_k):
    new_score_rank = x_corresponding
    err1 = err_computation(new_score_rank, top_k)
    the_index = np.arange(new_score_rank.shape[0])
    temp = the_index[j]
    the_index[j] = the_index[k]
    the_index[k] = temp
    new_score_rank = new_score_rank[the_index]
    err2 = err_computation(new_score_rank, top_k)

    return torch.abs(err1 - err2)


def avg_err(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):
    the_maxs, _ = torch.max(x_corresponding, 1)
    the_maxs = the_maxs.reshape(the_maxs.shape[0], 1).repeat(1, x_corresponding.shape[1])
    c = 2 * torch.ones_like(x_corresponding)
    x_corresponding = ( c.pow(x_corresponding) - 1) / c.pow(the_maxs)
    the_ones = torch.ones_like(x_corresponding)
    new_x_corresponding = torch.cat((the_ones, 1 - x_corresponding), 1)

    for i in range(x_corresponding.shape[1] - 1):
        x_corresponding = torch.mul(x_corresponding, new_x_corresponding[:, -x_corresponding.shape[1] - 1 - i : -1 - i])
    the_range = torch.arange(0., x_corresponding.shape[1]).repeat(x_corresponding.shape[0], 1) + 1
    score_rank = (1 / the_range[:, 0:]) * x_corresponding[:, 0:]
    final = torch.mean(torch.sum(score_rank, axis=1))
    print("Now Average ERR@k = ", final.item())

    return final.item()



def avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):
    c = 2 * torch.ones_like(x_sorted_scores[:, :top_k])
    numerator = c.pow(x_sorted_scores[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(x_sorted_scores[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    idcg = torch.sum((numerator / denominator), 1)
    new_score_rank = torch.zeros(y_ranks.shape[0], y_ranks[:, :top_k].shape[1])
    numerator = c.pow(x_corresponding.cuda()[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(new_score_rank[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    ndcg_list = torch.sum((numerator / denominator), 1) / idcg
    avg_ndcg = torch.mean(ndcg_list)
    print("Now Average NDCG@k = ", avg_ndcg.item())

    return avg_ndcg.item()








def lambdas_computation(x_similarity, y_similarity, top_k):
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = max_num * torch.ones_like(x_similarity[0, :])
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = max_num * torch.ones_like(y_similarity[0, :])

    # ***************************** ranking ******************************
    (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = torch.arange(y_similarity.shape[0]).view(y_similarity.shape[0], 1).repeat(1, y_similarity.shape[0])
    y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()

    # ***************************** pairwise delta ******************************
    sigma_tuned = sigma_1
    length_of_k = k_para * top_k
    y_sorted_scores = y_sorted_scores[:, 1 :(length_of_k + 1)]
    y_sorted_idxs = y_sorted_idxs[:, 1 :(length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1 :(length_of_k + 1)]
    pairs_delta = torch.zeros(y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0])

    for i in range(y_sorted_scores.shape[0]):
        # print(i/y_sorted_scores.shape[0])
        pairs_delta[:, :, i] = y_sorted_scores[i, :].view(y_sorted_scores.shape[1], 1) - y_sorted_scores[i, :].float()

    fraction_1 = - sigma_tuned / (1 + (pairs_delta * sigma_tuned).exp())
    x_delta = torch.zeros(y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0])
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k)

    for i in range(x_corresponding.shape[0]):
        x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]

    for i in range(x_corresponding.shape[0]):
        # print(i / x_corresponding.shape[0])
        x_delta[:, :, i] = x_corresponding[i, :].view(x_corresponding.shape[1], 1) - x_corresponding[i, :].float()

    S_x = torch.sign(x_delta)
    zero = torch.zeros_like(S_x)
    S_x = torch.where(S_x < 0, zero, S_x)

    # ***************************** NDCG delta from ranking ******************************
    ndcg_delta = torch.zeros(x_corresponding.shape[1], x_corresponding.shape[1], x_corresponding.shape[0])
    for i in range(y_similarity.shape[0]):
        # if i >= 0.4 * y_similarity.shape[0]:
        #     break
        for j in range(x_corresponding.shape[1]):
            for k in range(x_corresponding.shape[1]):
                if S_x[j, k, i] == 0:
                    continue
                if j < k:
                    the_delta = err_exchange_abs(x_corresponding[i, :], j, k, top_k)
                    # print(the_delta)
                    ndcg_delta[j, k, i] = the_delta
                    ndcg_delta[k, j, i] = the_delta

    without_zero = S_x * fraction_1 * ndcg_delta
    lambdas = torch.zeros(x_corresponding.shape[0], x_corresponding.shape[1])

    for i in range(lambdas.shape[0]):
        for j in range(lambdas.shape[1]):
            lambdas[i, j] = torch.sum(without_zero[j, :, i]) - torch.sum(without_zero[:, j, i])   # 本来是 -

    mid = torch.zeros_like(x_similarity)
    the_x = torch.arange(x_similarity.shape[0]).repeat(length_of_k, 1).transpose(0, 1).reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    the_y = y_sorted_idxs.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    the_data = lambdas.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    mid.index_put_((the_x, the_y.long()), the_data.cuda())


    return mid, x_sorted_scores, y_sorted_idxs, x_corresponding




def lambdas_computation_only_review(x_similarity, y_similarity, top_k):
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = max_num * torch.ones_like(x_similarity[0, :])
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = max_num * torch.ones_like(y_similarity[0, :])
    (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = torch.arange(y_similarity.shape[0]).view(y_similarity.shape[0], 1).repeat(1, y_similarity.shape[0])
    y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()
    length_of_k = k_para * top_k - 1
    y_sorted_idxs = y_sorted_idxs[:, 1 :(length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1 :(length_of_k + 1)]
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k)

    for i in range(x_corresponding.shape[0]):
        x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]

    return x_sorted_scores, y_sorted_idxs, x_corresponding




def train_fair(epoch, model_name, adj, output):
    model.train()
    y_similarity1 = simi(output[idx_train])
    x_similarity = simi(features[idx_train])
    lambdas1, x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation(x_similarity, y_similarity1, top_k)
    assert lambdas1.shape == y_similarity1.shape

    y_similarity = simi(output[idx_test])
    x_similarity = simi(features[idx_test])

    print("Ranking optimizing... ")
    x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation_only_review(x_similarity, y_similarity, top_k)
    all_ndcg_list_test.append(avg_err(x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, top_k))

    y_similarity1.backward(lambdas_para * lambdas1)
    optimizer.step()





def train(epoch, model_name, adj, flag=0):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    if model_name == 'SGC':
        output1 = model(features)
    else:
        output1 = model(features, adj)

    loss_train = F.cross_entropy(output1[idx_train], labels[idx_train])
    acc_train = accuracy(output1[idx_train], labels[idx_train])
    if flag == 0:
        loss_train.backward(retain_graph=True)
    else:
        loss_train.backward()
        optimizer.step()

    model.eval()
    if model_name == 'SGC':
        output = model(features)
    else:
        output = model(features, adj)


    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return output1



def test():
    model.eval()
    if model_name == 'SGC':
        output = model(features)
    else:
        output = model(features, adj)

    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()


dataset = "ACM"  # ["ACM", "coauthor-cs", "coauthor-phy"]
model_name = "SGC"  # ['SGC', 'GCN']
all_ndcg_list_test = []
lambdas_para = 1
k_para = 1
sigma_1 = -1
top_k = 10
pre_train = 0

if model_name == "GCN":
    if dataset == "ACM":
        adj, features, labels, idx_train, idx_val, idx_test, _ = load_data2("ACM")
        sigma_1 = 1e-3
        args.epochs = 200
    else:
        adj, features, labels, idx_train, idx_val, idx_test, _ = load_npz(dataset)
        if dataset == "coauthor-cs":
            sigma_1 = 3e-3
            args.epochs = 300
        else:
            sigma_1 = 4e-3
            args.epochs = 600
else:
    if dataset == "ACM":
        adj, features, labels, idx_train, idx_val, idx_test, _ = load_data2("ACM")
        sigma_1 = 2e-2
        args.epochs = 15
        pre_train = 300
    else:
        adj, features, labels, idx_train, idx_val, idx_test, _ = load_npz(dataset)
        if dataset == "coauthor-cs":
            sigma_1 = 1e-2
            args.epochs = 40
            pre_train = 500
        else:
            sigma_1 = 3e-2
            args.epochs = 15
            pre_train = 500


print("Using {} dataset".format(dataset))

model = get_model(model_name, features.size(1), labels.max().item() + 1, args.hidden, args.dropout,
                  args.cuda)

optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)

if args.model == "SGC":
    features, precompute_time = sgc_precompute(features, adj, args.degree)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

for epoch in range(pre_train):
    output = train(epoch, model_name, adj, 1)

for epoch in range(args.epochs):
    output = train(epoch, model_name, adj)
    train_fair(epoch, model_name, adj, output)

test()

