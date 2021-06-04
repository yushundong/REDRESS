import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA
from numpy import inf
import scipy.io as scio
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import pickle as pkl
import networkx as nx
from sklearn.preprocessing import OneHotEncoder

pca = PCA(n_components=200)



def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize2(mx):
    """Row-normalize sparse matrix"""
    rowmode = np.sqrt(np.array(mx.power(2).sum(1)))
    r_inv = np.power(rowmode, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data_news(dataset="amazon_electronics_computers"):

    print(dataset)



    # edge
    data = pd.read_csv('data/' + dataset + '/edge_list.edg', header=None)
    edges = data.values.tolist()

    edges = [[int(edge[0]), int(edge[1])] for edge in edges]

    graph = nx.from_edgelist(edges)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    adj = np.array(nx.adjacency_matrix(graph).todense())
    adj = normalize(adj + sp.eye(adj.shape[0]))

    print("A Size  " + str(adj.shape))




    # feature

    features = np.array(pd.read_csv('data/' + dataset + '/features.csv', header=None))
    features = normalize(sp.coo_matrix(features))

    print("feature Size  " + str(features.shape))


    # labels

    labels = np.array(pd.read_csv('data/' + dataset + '/labels.csv', header=None))

    print("Label Size  " + str(labels.shape))



    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)


    # split


    node_perm = np.random.permutation(labels.shape[0])

    num_train = int(0.1 * adj.shape[0])
    num_val = int(0.2 * adj.shape[0])

    idx_train = node_perm[:num_train]
    idx_val = node_perm[num_train:num_train + num_val]
    idx_test = node_perm[num_train + num_val:]

    a = pca.fit_transform(np.array(features.todense()))

    features = torch.FloatTensor(a)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test





def load_data_with_ini_adj(dataset_source):
    print(dataset_source)
    data = sio.loadmat("../data/{}.mat".format(dataset_source))
    features = data["Attributes"]
    adj = data["Network"]
    labels = data["Label"]

    ini_adj = adj

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # features = preprocessing.normalize(features, norm='l2', axis=0)

    node_perm = np.random.permutation(labels.shape[0])
    num_train = int(0.05 * adj.shape[0])
    num_val = int(0.1 * adj.shape[0])

    idx_train = node_perm[:num_train]
    idx_val = node_perm[num_train:num_train + num_val]
    idx_test = node_perm[num_train + num_val:]

    a = pca.fit_transform(np.array(features.todense()))

    features = torch.FloatTensor(a)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, ini_adj



def load_data2(dataset_source):
    print(dataset_source)
    dataFile = 'data/' + dataset_source + '.mat'
    data = scio.loadmat(dataFile)
    features = data["Attributes"]
    adj = data["Network"]
    labels = data["Label"]

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    adj_ori = adj
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # features = preprocessing.normalize(features, norm='l2', axis=0)

    node_perm = np.random.permutation(labels.shape[0])
    num_train = int(0.05 * adj.shape[0])
    num_val = int(0.1 * adj.shape[0])
    # num_train = int(0.1 * adj.shape[0])
    # num_val = int(0.2 * adj.shape[0])
    # num_train = int(0.2 * adj.shape[0])
    # num_val = int(0.3 * adj.shape[0])
    idx_train = node_perm[:num_train]
    idx_val = node_perm[num_train:num_train + num_val]
    idx_test = node_perm[num_train + num_val:]

    a = pca.fit_transform(np.array(features.todense()))
    simi = normalize2(sp.csr_matrix(features[idx_train, :], dtype=np.float32)).dot(normalize2(sp.csr_matrix(features[idx_train, :], dtype=np.float32)).T)
    d = np.array(simi.sum(axis=1).flatten())[0]
    l_s = np.diag(d) - simi

    features = torch.FloatTensor(a)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, l_s, adj_ori



def sparse_to_tuple(sparse_matrix):
    if not sp.isspmatrix_coo(sparse_matrix):
        sparse_matrix = sparse_matrix.tocoo()
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col)).transpose()
    values = sparse_matrix.data
    shape = sparse_matrix.shape
    return indices, values, shape


def load_npz(dataset):
    file_map = {'coauthor-cs': 'ms_academic_cs.npz', 'coauthor-phy': 'ms_academic_phy.npz'}
    file_name = file_map[dataset]

    print(dataset)

    with np.load('data/' + file_name, allow_pickle=True) as f:
        f = dict(f)
        features = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']), shape=f['attr_shape'])
        features = features.astype(np.float64)
        features = normalize_features(features)

        labels = f['labels'].reshape(-1, 1)
        labels = OneHotEncoder(sparse=False).fit_transform(labels)

        adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']), shape=f['adj_shape'])
        adj_ori = adj
        adj = normalize(adj + sp.eye(adj.shape[0]))

    print('Total size : ', str(labels.shape[0]))




    node_perm = np.random.permutation(labels.shape[0])
    num_train = int(0.05 * labels.shape[0])
    num_val = int(0.1 * labels.shape[0])

    idx_train = node_perm[:num_train]
    idx_val = node_perm[num_train:num_train + num_val]
    idx_test = node_perm[num_train + num_val:]
    if dataset == 'coauthor-phy':
        idx_test = node_perm[num_train + num_val:int(0.6 * labels.shape[0])]  # 0.6

    a = pca.fit_transform(np.array(features.todense()))
    simi = normalize2(sp.csr_matrix(features[idx_train, :], dtype=np.float32)).dot(normalize2(sp.csr_matrix(features[idx_train, :], dtype=np.float32)).T)
    d = np.array(simi.sum(axis=1).flatten())[0]
    l_s = np.diag(d) - simi

    features = sparse_mx_to_torch_sparse_tensor(features).to_dense()
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, l_s, adj_ori






def load_data_new_1(dataset="BlogCatalog", mode='s'):
    print('Loading {} dataset...'.format(dataset))
    dataFile = '../data/' + dataset + '.mat'
    data = scio.loadmat(dataFile)
    labels = encode_onehot(list(data['Label'][:, 0]))
    # adj = sp.csr_matrix(data['Network'].toarray()[:, :])
    adj = data["Network"]


    features = data['Attributes'].toarray()[:, :]
    print('Dataset has {} nodes, {} features.'.format(adj.shape[0], features.shape[1]))

    adj = normalize(adj + sp.eye(adj.shape[0]))

    list_split = []
    length_of_data = adj.shape[0]
    train_percent = 0.1
    val_percent = 0.2

    for i in range(length_of_data):
        list_split.append(i)

    node_perm = np.random.permutation(labels.shape[0])
    idx_train = node_perm[:int(train_percent * length_of_data)]  # list_split
    idx_val = node_perm[int(train_percent * length_of_data): int(train_percent * length_of_data + val_percent * length_of_data)]
    idx_test = node_perm[int(train_percent * length_of_data + val_percent * length_of_data): ]

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    with np.errstate(divide='ignore'):
        r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def load_data_new(dataset_str, mode='r'):  # {'pubmed', 'citeseer', 'cora'}

    print("Loading dataset:  " + dataset_str)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + sp.eye(adj.shape[0])
    D = []
    for i in range(adj.sum(axis=1).shape[0]):
        D.append(adj.sum(axis=1)[i, 0])
    D = np.diag(D)
    l = adj
    if mode == 'r':
        with np.errstate(divide='ignore'):
            D_norm = np.linalg.inv(D)
        adj = sp.coo_matrix(D_norm.dot(l))
    elif mode == 's':
        with np.errstate(divide='ignore'):
            D_norm = D ** (-0.5)
        D_norm[D_norm == inf] = 0
        adj = sp.coo_matrix(D_norm.dot(l).dot(D_norm))

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    labels = torch.LongTensor(np.where(labels)[1])

    idx_test = torch.LongTensor(test_idx_range.tolist())
    idx_train = torch.LongTensor(range(len(y)))
    idx_val = torch.LongTensor(range(len(y), len(y) + 500))

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data_new2(dataset_str):  # {'pubmed', 'citeseer', 'cora'}

    print("Loading dataset:  " + dataset_str)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))



    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = normalize(adj + sp.eye(adj.shape[0]))

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    labels = torch.LongTensor(np.where(labels)[1])

    idx_test = torch.LongTensor(test_idx_range.tolist())
    idx_train = torch.LongTensor(range(len(y)))
    idx_val = torch.LongTensor(range(len(y), len(y) + 500))

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
