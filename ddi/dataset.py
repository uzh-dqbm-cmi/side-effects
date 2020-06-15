import os
import numpy as np
import torch
from .utilities import ModelScore, ReaderWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight


class DDIDataTensor(Dataset):

    def __init__(self, X_feat, y):
        self.X_feat = X_feat  # tensor.float32, (drug pairs, features)
        # drug interactions
        self.y = y  # tensor.float32, (drug pairs,)
        self.num_samples = self.y.size(0)  # int, number of drug pairs

    def __getitem__(self, indx):

        return(self.X_feat[indx], self.y[indx], indx)

    def __len__(self):
        return(self.num_samples)


class PartitionDataTensor(Dataset):

    def __init__(self, ddi_datatensor, partition_ids, dsettype, fold_num):
        self.ddi_datatensor = ddi_datatensor  # instance of :class:`DDIDataTensor`
        self.partition_ids = partition_ids  # list of indices for drug pairs
        self.dsettype = dsettype  # string, dataset type (i.e. train, validation, test)
        self.fold_num = fold_num  # int, fold number
        self.num_samples = len(self.partition_ids)  # int, number of docs in the partition

    def __getitem__(self, indx):
        target_id = self.partition_ids[indx]
        return self.ddi_datatensor[target_id]

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

def preprocess_features(feat_fpath):
    X_fea = np.loadtxt(feat_fpath,dtype=float,delimiter=",")
    r, c = np.triu_indices(len(X_fea),1) # take indices off the diagnoal by 1
    return np.concatenate((X_fea[r], X_fea[c]), axis=1)

def preprocess_labels(interaction_fpath):
    interaction_matrix = np.loadtxt(interaction_fpath,dtype=float,delimiter=",")
    r, c = np.triu_indices(len(interaction_matrix),1) # take indices off the diagnoal by 1
    return interaction_matrix[r,c]

def create_setvector_features(X, num_sim_types):
    """reshape concatenated features from every similarity type matrix into set of vectors per ddi example"""
    e = X[np.newaxis, :, :]
    f = np.transpose(e, axes=(0, 2, 1))
    splitter = 2*num_sim_types 
    g = np.concatenate(np.split(f, splitter, axis=1), axis=0)
    h = np.transpose(g, axes=(2,0, 1))
    return h

def get_stratified_partitions(ddi_datatensor, num_folds=5, random_state=42):
    """Generate 5-fold stratified sample of drug-pair ids based on the interaction label

    Args:
        ddi_datatensor: instance of :class:`DDIDataTensor`
    """
    skf_trte = StratifiedKFold(n_splits=num_folds, random_state=random_state, shuffle=True)  # split train and test
    data_partitions = {}
    X = ddi_datatensor.X_feat
    y = ddi_datatensor.y
    fold_num = 0
    for train_index, test_index in skf_trte.split(X,y):
    
        data_partitions[fold_num] = {'train': train_index,
                                     'test': test_index}
        print("fold_num:", fold_num)
        print('train data')
        report_label_distrib(y[train_index])
        print('test data')
        report_label_distrib(y[test_index])
        print()
        fold_num += 1
        print("-"*25)
    return(data_partitions)

def get_validation_partitions(ddi_datatensor, num_folds=2, valid_set_portion=0.1, random_state=42):
    """Generate stratified train/validation split of drug-pair ids based on the interaction label

    Args:
        ddi_datatensor: instance of :class:`DDIDataTensor`
    """
    skf_trte = StratifiedShuffleSplit(n_splits=num_folds, 
                                      test_size=valid_set_portion, 
                                      random_state=random_state)  # split train and test
    data_partitions = {}
    X = ddi_datatensor.X_feat
    y = ddi_datatensor.y
    fold_num = 0
    for train_index, test_index in skf_trte.split(X,y):
    
        data_partitions[fold_num] = {'train': train_index,
                                     'validation': test_index}
        print("fold_num:", fold_num)
        print('train data')
        report_label_distrib(y[train_index])
        print('validation data')
        report_label_distrib(y[test_index])
        print()
        fold_num += 1
        print("-"*25)
    return(data_partitions)

def report_label_distrib(labels):
    classes, counts = np.unique(labels, return_counts=True)
    norm_counts = counts/counts.sum()
    for i, label in enumerate(classes):
        print("class:", label, "norm count:", norm_counts[i])


def validate_partitions(data_partitions, drugpairs_ids, valid_set_portion=0.1, test_set_portion=0.2):
    if(not isinstance(drugpairs_ids, set)):
        drugpairs_ids = set(drugpairs_ids)
    num_pairs = len(drugpairs_ids)
    test_set_accum = set([])
    for fold_num in data_partitions:
        print('fold_num', fold_num)
        tr_ids = data_partitions[fold_num]['train']
        te_ids = data_partitions[fold_num]['test']

        tr_te = set(tr_ids).intersection(te_ids)
        # assert there is no overlap among train and test partition within a fold
        assert len(tr_te) == 0
        print('expected test set size:', test_set_portion*num_pairs, '; actual test set size:', len(te_ids))
        print()
        assert np.abs(test_set_portion*num_pairs - len(te_ids)) <= 2
        test_set_accum = test_set_accum.union(te_ids)
    # verify that assembling test sets from each of the five folds would be equivalent to all drugpair ids
    assert len(test_set_accum) == num_pairs
    assert test_set_accum == drugpairs_ids
    print("passed intersection and overlap test (i.e. train, validation and test sets are not",
          "intersecting in each fold and the concatenation of test sets from each fold is",
          "equivalent to the whole dataset)")


def generate_partition_datatensor(ddi_datatensor, data_partitions):
    datatensor_partitions = {}
    for fold_num in data_partitions:
        datatensor_partitions[fold_num] = {}
        for dsettype in data_partitions[fold_num]:
            target_ids = data_partitions[fold_num][dsettype]
            datatensor_partition = PartitionDataTensor(ddi_datatensor, target_ids, dsettype, fold_num)
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
    class_weights = compute_class_weight('balanced', classes, labels_tensor.numpy())
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
