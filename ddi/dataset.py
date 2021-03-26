import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import norm as scpnorm
import pandas as pd
from .utilities import ModelScore, ReaderWriter


class DDIDataTensor(Dataset):

    def __init__(self, y, X_a, X_b):
          
        self.X_a = X_a # tensor.float32, (drug pairs, features)
        self.X_b = X_b # tensor.float32, (drug pairs, features)
        
        # drug interactions
        self.y = y  # tensor.float32, (drug pairs,)
        self.num_samples = self.y.size(0)  # int, number of drug pairs

    def __getitem__(self, indx):

        return(self.X_a[indx], self.X_b[indx], self.y[indx], indx)

    def __len__(self):
        return(self.num_samples)

class GIPDataTensor(Dataset):

    def __init__(self, X_a, X_b):
        self.X_a = X_a # tensor.float32, (drug pairs, gip features)
        self.X_b = X_b # tensor.float32, (drug pairs, gip features)
        # drug interactions
        self.num_samples = self.X_a.size(0)  # int, number of drug pairs

    def __getitem__(self, indx):

        return(self.X_a[indx], self.X_b[indx], indx)

    def __len__(self):
        return(self.num_samples)


class PartitionDataTensor(Dataset):

    def __init__(self, ddi_datatensor, gip_datatensor, partition_ids, dsettype, fold_num, is_siamese):
        self.ddi_datatensor = ddi_datatensor  # instance of :class:`DDIDataTensor`
        self.gip_datatensor = gip_datatensor # instance of :class:`GIPDataTensor`
        self.partition_ids = partition_ids  # list of indices for drug pairs
        self.dsettype = dsettype  # string, dataset type (i.e. train, validation, test)
        self.fold_num = fold_num  # int, fold number
        self.num_samples = len(self.partition_ids)  # int, number of docs in the partition
        self.is_siamese = is_siamese

    def __getitem__(self, indx):
        target_id = self.partition_ids[indx]
        X_a_gip, X_b_gip, gip_indx = self.gip_datatensor[target_id]
        # combine gip with other matrices
        X_a, X_b, y, ddi_indx = self.ddi_datatensor[target_id]
        X_a_comb = torch.cat([X_a, X_a_gip], axis=0)
        X_b_comb = torch.cat([X_b, X_b_gip], axis=0)
        X_comb = torch.cat([X_a_comb, X_b_comb])#.view(-1)
        
        if (self.is_siamese):
            return X_a_comb, X_b_comb, y, ddi_indx
        else:
            return X_comb, y, ddi_indx
        
    def __len__(self):
        return(self.num_samples)

def construct_load_dataloaders(dataset_fold, dsettypes, config, wrk_dir):
    """construct dataloaders for the dataset for one fold

       Args:
            dataset_fold: dictionary,
                          example: {'train': <neural.dataset.PartitionDataTensor at 0x1cec95c96a0>,
                                    'validation': <neural.dataset.PartitionDataTensor at 0x1cec95c9208>,
                                    'test': <neural.dataset.PartitionDataTensor at 0x1cec95c9240>,
                                    'class_weights': tensor([0.6957, 1.7778])
                                   }
            dsettype: list, ['train', 'validation', 'test']
            config: dict, {'batch_size': int, 'num_workers': int}
            wrk_dir: string, folder path
    """

    # setup data loaders
    data_loaders = {}
    epoch_loss_avgbatch = {}
    flog_out = {}
    score_dict = {}
    class_weights = {}
    for dsettype in dsettypes:
        if(dsettype == 'train'):
            shuffle = True
            class_weights[dsettype] = dataset_fold['class_weights']
        else:
            shuffle = False
            class_weights[dsettype] = None
        data_loaders[dsettype] = DataLoader(dataset_fold[dsettype],
                                            batch_size=config['batch_size'],
                                            shuffle=shuffle,
                                            num_workers=config['num_workers'])

        epoch_loss_avgbatch[dsettype] = []
        score_dict[dsettype] = ModelScore(0, 0.0, 0.0, 0.0, 0.0, 0.0)  # (best_epoch, auc, aupr, f1, precision, recall)
        if(wrk_dir):
            flog_out[dsettype] = os.path.join(wrk_dir, dsettype + ".log")
        else:
            flog_out[dsettype] = None

    return (data_loaders, epoch_loss_avgbatch, score_dict, class_weights, flog_out)

def preprocess_features(feat_fpath, dsetname, fill_diag = None):
    if dsetname in {'DS1', 'DS3'}:
        X_fea = np.loadtxt(feat_fpath,dtype=float,delimiter=",")
    elif dsetname == 'DS2':
        X_fea = pd.read_csv(feat_fpath).values[:,1:]
    X_fea = X_fea.astype(np.float32)
    if fill_diag is not None:
        np.fill_diagonal(X_fea, fill_diag)
    return get_features_from_simmatrix(X_fea)

def get_features_from_simmatrix(sim_mat):
    """
    Args:
        sim_mat: np.array, mxm (drug pair similarity matrix)
    """
    r, c = np.triu_indices(len(sim_mat),1) # take indices off the diagnoal by 1
    return np.concatenate((sim_mat[r], sim_mat[c], sim_mat[r,c].reshape(-1,1), sim_mat[c,r].reshape(-1,1)), axis=1)

def preprocess_labels(interaction_fpath, dsetname):
    interaction_mat = get_interaction_mat(interaction_fpath, dsetname)
    return get_y_from_interactionmat(interaction_mat)

def get_y_from_interactionmat(interaction_mat):
    r, c = np.triu_indices(len(interaction_mat),1) # take indices off the diagnoal by 1
    return interaction_mat[r,c]

def compute_gip_profile(adj, bw=1.):
    """approach based on Olayan et al. https://doi.org/10.1093/bioinformatics/btx731 """
    
    ga = np.dot(adj,np.transpose(adj))
    ga = bw*ga/np.mean(np.diag(ga))
    di = np.diag(ga)
    x =  np.tile(di,(1,di.shape[0])).reshape(di.shape[0],di.shape[0])
    d =x+np.transpose(x)-2*ga
    return np.exp(-d)

def compute_kernel(mat, k_bandwidth, epsilon=1e-9):
    """computes gaussian kernel from 2D matrix
    
       Approach based on van Laarhoven et al. doi:10.1093/bioinformatics/btr500
    
    """
    r, c = mat.shape # 2D matrix
    # computes pairwise l2 distance
    dist_kernel = squareform(pdist(mat, metric='euclidean')**2)
    gamma = k_bandwidth/(np.clip((scpnorm(mat, axis=1, keepdims=True)**2) * 1/c, a_min=epsilon, a_max=None))
    return np.exp(-gamma*dist_kernel)

def construct_sampleid_ddipairs(interaction_mat):
    # take indices off the diagnoal by 1
    r, c = np.triu_indices(len(interaction_mat),1)
    sid_ddipairs = {sid:ddi_pair for sid, ddi_pair in enumerate(zip(r,c))}
    return sid_ddipairs

def get_num_drugs(interaction_fpath, dsetname):
    if dsetname in {'DS1', 'DS3'}:
        interaction_matrix = np.loadtxt(interaction_fpath,dtype=float,delimiter=",")
    elif dsetname == 'DS2':
        interaction_matrix = pd.read_csv(interaction_fpath).values[:,1:]
    return interaction_matrix.shape[0]

def get_interaction_mat(interaction_fpath, dsetname):
    if dsetname in {'DS1', 'DS3'}:
        interaction_matrix = np.loadtxt(interaction_fpath,dtype=float,delimiter=",")
    elif dsetname == 'DS2':
        interaction_matrix = pd.read_csv(interaction_fpath).values[:,1:]
    return interaction_matrix.astype(np.int32)

def get_similarity_matrix(feat_fpath, dsetname):
    if dsetname in {'DS1', 'DS3'}:
        X_fea = np.loadtxt(feat_fpath,dtype=float,delimiter=",")
    elif dsetname == 'DS2':
        X_fea = pd.read_csv(feat_fpath).values[:,1:]
    X_fea = X_fea.astype(np.float32)
    return X_fea

def create_setvector_features(X, num_sim_types):
    """reshape concatenated features from every similarity type matrix into set of vectors per ddi example"""
    e = X[np.newaxis, :, :]
    f = np.transpose(e, axes=(0, 2, 1))
    splitter = num_sim_types 
    g = np.concatenate(np.split(f, splitter, axis=1), axis=0)
    h = np.transpose(g, axes=(2,0, 1))
    return h

def get_stratified_partitions(y, num_folds=5, valid_set_portion=0.1, random_state=42):
    """Generate 5-fold stratified sample of drug-pair ids based on the interaction label

    Args:
        y: ddi labels
    """
    skf_trte = StratifiedKFold(n_splits=num_folds, random_state=random_state, shuffle=True)  # split train and test
    
    skf_trv = StratifiedShuffleSplit(n_splits=2, 
                                     test_size=valid_set_portion, 
                                     random_state=random_state)  # split train and test
    data_partitions = {}
    X = np.zeros(len(y))
    fold_num = 0
    for train_index, test_index in skf_trte.split(X,y):
        
        x_tr = np.zeros(len(train_index))
        y_tr = y[train_index]

        for tr_index, val_index in skf_trv.split(x_tr, y_tr):
            tr_ids = train_index[tr_index]
            val_ids = train_index[val_index]
            data_partitions[fold_num] = {'train': tr_ids,
                                         'validation': val_ids,
                                         'test': test_index}
            
        print("fold_num:", fold_num)
        print('train data')
        report_label_distrib(y[tr_ids])
        print('validation data')
        report_label_distrib(y[val_ids])
        print('test data')
        report_label_distrib(y[test_index])
        print()
        fold_num += 1
        print("-"*25)
    return(data_partitions)

def validate_partitions(data_partitions, drugpairs_ids, valid_set_portion=0.1, test_set_portion=0.2):
    if(not isinstance(drugpairs_ids, set)):
        drugpairs_ids = set(drugpairs_ids)
    num_pairs = len(drugpairs_ids)
    test_set_accum = set([])
    for fold_num in data_partitions:
        print('fold_num', fold_num)
        tr_ids = data_partitions[fold_num]['train']
        val_ids = data_partitions[fold_num]['validation']
        te_ids = data_partitions[fold_num]['test']

        tr_val = set(tr_ids).intersection(val_ids)
        tr_te = set(tr_ids).intersection(te_ids)
        te_val = set(te_ids).intersection(val_ids)
        
        tr_size = len(tr_ids) + len(val_ids)
        # assert there is no overlap among train and test partition within a fold
        print('expected validation set size:', valid_set_portion*tr_size, '; actual test set size:', len(val_ids))
        assert len(tr_te) == 0
        print('expected test set size:', test_set_portion*num_pairs, '; actual test set size:', len(te_ids))
        print()
        assert np.abs(valid_set_portion*tr_size - len(val_ids)) <= 2
        assert np.abs(test_set_portion*num_pairs - len(te_ids)) <= 2
        for s in (tr_val, tr_te, te_val):
            assert len(s) == 0
        s_union = set(tr_ids).union(val_ids).union(te_ids)
        assert len(s_union) == num_pairs
        test_set_accum = test_set_accum.union(te_ids)
    # verify that assembling test sets from each of the five folds would be equivalent to all drugpair ids
    assert len(test_set_accum) == num_pairs
    assert test_set_accum == drugpairs_ids
    print("passed intersection and overlap test (i.e. train, validation and test sets are not",
          "intersecting in each fold and the concatenation of test sets from each fold is",
          "equivalent to the whole dataset)")

def report_label_distrib(labels):
    classes, counts = np.unique(labels, return_counts=True)
    norm_counts = counts/counts.sum()
    for i, label in enumerate(classes):
        print("class:", label, "norm count:", norm_counts[i])


def generate_partition_datatensor(ddi_datatensor, gip_dtensor_perfold, data_partitions, is_siamese):
    datatensor_partitions = {}
    for fold_num in data_partitions:
        datatensor_partitions[fold_num] = {}
        gip_datatensor = gip_dtensor_perfold[fold_num]
        for dsettype in data_partitions[fold_num]:
            target_ids = data_partitions[fold_num][dsettype]
            datatensor_partition = PartitionDataTensor(ddi_datatensor, gip_datatensor, target_ids, dsettype, fold_num, is_siamese)
            datatensor_partitions[fold_num][dsettype] = datatensor_partition
    compute_class_weights_per_fold_(datatensor_partitions)

    return(datatensor_partitions)

def build_datatensor_partitions(data_partitions, ddi_datatensor):
    datatensor_partitions = generate_partition_datatensor(ddi_datatensor, data_partitions)
    compute_class_weights_per_fold_(datatensor_partitions)
    return datatensor_partitions

def compute_class_weights(labels_tensor):
    classes, counts = np.unique(labels_tensor, return_counts=True)
    # print("classes", classes)
    # print("counts", counts)
    class_weights = compute_class_weight('balanced', classes=classes, y=labels_tensor.numpy())
    return class_weights


def compute_class_weights_per_fold_(datatensor_partitions):
    """computes inverse class weights and updates the passed dictionary

    Args:
        datatensor_partitions: dictionary, {fold_num, int: {datasettype, string:{datapartition, instance of
        :class:`PartitionDataTensor`}}}}

    Example:
        datatensor_partitions
            {0: {'train': <neural.dataset.PartitionDataTensor at 0x1cec95c96a0>,
                 'validation': <neural.dataset.PartitionDataTensor at 0x1cec95c9208>,
                 'test': <neural.dataset.PartitionDataTensor at 0x1cec95c9240>
                }, ..
            }
        is updated after computation of class weights to
            {0: {'train': <neural.dataset.PartitionDataTensor at 0x1cec95c96a0>,
                 'validation': <neural.dataset.PartitionDataTensor at 0x1cec95c9208>,
                 'test': <neural.dataset.PartitionDataTensor at 0x1cec95c9240>,
                 'class_weights': tensor([0.6957, 1.7778]),
                 }, ..
            }
    """

    for fold_num in datatensor_partitions:  # looping over the numbered folds
        dpartition = datatensor_partitions[fold_num]['train']
        partition_ids = dpartition.partition_ids
        labels = dpartition.ddi_datatensor.y[partition_ids]
        datatensor_partitions[fold_num]['class_weights'] = torch.from_numpy(compute_class_weights(labels)).float()

def read_pickles(data_dir, device):

    # Read stored data structures
    data_partitions = ReaderWriter.read_data(os.path.join(data_dir, 'data_partitions.pkl'))
    # instance of :class:`DDIDataTensor`
    ddi_datatensor = ReaderWriter.read_tensor(os.path.join(data_dir, 'ddi_datatensor.torch'), device)

    return data_partitions, ddi_datatensor