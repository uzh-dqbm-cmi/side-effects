import os
import shutil
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_curve, precision_recall_curve, accuracy_score, \
                            recall_score, precision_score, roc_auc_score, auc, average_precision_score
from matplotlib import pyplot as plt
from os.path import dirname, abspath


class ModelScore:
    def __init__(self, best_epoch_indx, s_auc, s_aupr, s_f1, s_precision, s_recall):
        self.best_epoch_indx = best_epoch_indx
        self.s_auc = s_auc
        self.s_aupr = s_aupr
        self.s_f1 = s_f1
        self.s_precision = s_precision
        self.s_recall = s_recall


    def __repr__(self):
        desc = " best_epoch_indx:{}\n auc:{} \n apur:{} \n f1:{} \n precision:{} \n recall:{} \n" \
               "".format(self.best_epoch_indx, self.s_auc, self.s_aupr, self.s_f1, self.s_precision, self.s_recall)
        return desc

def get_performance_results(similarity_type, target_dir, num_folds, dsettype, suffix_testfname=None):
    all_perf = {}
    num_metrics = 3 # number of metrics to focus on
    perf_dict = [{} for i in range(num_metrics)]  # track auc, aupr, f1 measure
    if dsettype == 'train':
        prefix = 'train_val'
    else:
        prefix = dsettype
        if suffix_testfname:
            prefix = prefix + "_" + suffix_testfname

    for fold_num in range(num_folds):

        fold_dir = os.path.join(target_dir,
                '{}'.format(prefix),
                'fold_{}'.format(fold_num))
        # print('fold_dir:', fold_dir)

        score_file = os.path.join(fold_dir, 'score_{}.pkl'.format(dsettype))
        # print(score_file)

        if os.path.isfile(score_file):
            mscore = ReaderWriter.read_data(score_file)
            perf_dict[0]['fold{}'.format(fold_num)] = mscore.s_auc
            perf_dict[1]['fold{}'.format(fold_num)] = mscore.s_aupr
            perf_dict[2]['fold{}'.format(fold_num)] = mscore.s_f1
    perf_df = []
    for i in range(num_metrics):
        all_perf = perf_dict[i]
        all_perf_df = pd.DataFrame(all_perf, index=[similarity_type])
        median = all_perf_df.median(axis=1)
        mean = all_perf_df.mean(axis=1)
        stddev = all_perf_df.std(axis=1)
        all_perf_df['mean'] = mean
        all_perf_df['median'] = median
        all_perf_df['stddev'] = stddev
        perf_df.append(all_perf_df.sort_values('mean', ascending=False))
    return perf_df


def build_performance_dfs(similarity_types, target_dir, num_folds, dsettype, suffix_testfname=None):
    auc_df = pd.DataFrame()
    aupr_df = pd.DataFrame()
    f1_df = pd.DataFrame()
    target_dir = create_directory(target_dir, directory="parent")
    print(target_dir)
    for sim_type in similarity_types:
        if suffix_testfname is not None:
            suff_testfname = suffix_testfname + sim_type
        else:
            suff_testfname = None
        s_auc, s_aupr, s_f1 = get_performance_results(sim_type, 
                                                      target_dir, 
                                                      num_folds, 
                                                      dsettype, 
                                                      suffix_testfname=suff_testfname)
        auc_df = pd.concat([auc_df, s_auc], sort=True)
        aupr_df = pd.concat([aupr_df, s_aupr], sort=True)
        f1_df = pd.concat([f1_df, s_f1], sort=True)

    return auc_df, aupr_df, f1_df


class ReaderWriter(object):
    """class for dumping, reading and logging data"""
    def __init__(self):
        pass

    @staticmethod
    def dump_data(data, file_name, mode="wb"):
        """dump data by pickling
           Args:
               data: data to be pickled
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            pickle.dump(data, f)

    @staticmethod
    def read_data(file_name, mode="rb"):
        """read dumped/pickled data
           Args:
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            data = pickle.load(f)
        return(data)

    @staticmethod
    def dump_tensor(data, file_name):
        """
        Dump a tensor using PyTorch's custom serialization. Enables re-loading the tensor on a specific gpu later.
        Args:
            data: Tensor
            file_name: file path where data will be dumped
        Returns:
        """
        torch.save(data, file_name)

    @staticmethod
    def read_tensor(file_name, device):
        """read dumped/pickled data
           Args:
               file_name: file path where data will be dumped
               device: the gpu to load the tensor on to
        """
        data = torch.load(file_name, map_location=device)
        return data

    @staticmethod
    def write_log(line, outfile, mode="a"):
        """write data to a file
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(outfile, mode) as f:
            f.write(line)

    @staticmethod
    def read_log(file_name, mode="r"):
        """write data to a file
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(file_name, mode) as f:
            for line in f:
                yield line


def create_directory(folder_name, directory="current"):
    """create directory/folder (if it does not exist) and returns the path of the directory
       Args:
           folder_name: string representing the name of the folder to be created
       Keyword Arguments:
           directory: string representing the directory where to create the folder
                      if `current` then the folder will be created in the current directory
    """
    if directory == "current":
        path_current_dir = os.path.dirname(__file__)  # __file__ refers to utilities.py
    elif directory == "parent":
        path_current_dir = dirname(dirname(abspath(__file__)))
    else:
        path_current_dir = directory
    print("path_current_dir", path_current_dir)
        
    path_new_dir = os.path.normpath(os.path.join(path_current_dir, folder_name))
    if not os.path.exists(path_new_dir):
        os.makedirs(path_new_dir)
    return(path_new_dir)


def get_device(to_gpu, index=0):
    is_cuda = torch.cuda.is_available()
    if(is_cuda and to_gpu):
        target_device = 'cuda:{}'.format(index)
    else:
        target_device = 'cpu'
    return torch.device(target_device)


def report_available_cuda_devices():
    if(torch.cuda.is_available()):
        n_gpu = torch.cuda.device_count()
        print('number of GPUs available:', n_gpu)
        for i in range(n_gpu):
            print("cuda:{}, name:{}".format(i, torch.cuda.get_device_name(i)))
            device = torch.device('cuda', i)
            get_cuda_device_stats(device)
            print()
    else:
        print("no GPU devices available!!")

def get_cuda_device_stats(device):
    print('total memory available:', torch.cuda.get_device_properties(device).total_memory/(1024**3), 'GB')
    print('total memory allocated on device:', torch.cuda.memory_allocated(device)/(1024**3), 'GB')
    print('max memory allocated on device:', torch.cuda.max_memory_allocated(device)/(1024**3), 'GB')
    print('total memory cached on device:', torch.cuda.memory_cached(device)/(1024**3), 'GB')
    print('max memory cached  on device:', torch.cuda.max_memory_cached(device)/(1024**3), 'GB')

def get_interaction_stat(matrix):
    w, h = matrix.shape
    totalnum_elements = w*h
    nonzero_elem = np.count_nonzero(matrix)
    zero_elem = totalnum_elements - nonzero_elem
    print('number of rows: {}, cols: {}'.format(w, h))
    print('total number of elements', totalnum_elements)
    print('number of nonzero elements', nonzero_elem)
    print('number of zero elements', zero_elem)
    print('diagnoal elements ', np.diag(matrix))

def perfmetric_report(pred_target, ref_target, probscore, epoch, outlog):
    lsep = "\n"
    report = "Epoch: {}".format(epoch) + lsep
    report += "Classification report on all events:" + lsep
    report += str(classification_report(ref_target, pred_target)) + lsep
    report += "macro f1:" + lsep
    macro_f1 = f1_score(ref_target, pred_target, average='macro')
    report += str(macro_f1) + lsep
    report += "micro f1:" + lsep
    micro_f1 = f1_score(ref_target, pred_target, average='micro')
    report += str(micro_f1) + lsep
    report += "accuracy:" + lsep
    accuracy = accuracy_score(ref_target, pred_target)
    report += str(accuracy) + lsep
        
    s_auc = roc_auc_score(ref_target, probscore)
    report += "AUC:\n" + str(s_auc) + lsep
    precision_scores, recall_scores, __ = precision_recall_curve(ref_target, probscore)
    s_aupr = auc(recall_scores, precision_scores)
    report += "AUPR:\n" + str(s_aupr) + lsep
    s_f1 = f1_score(ref_target, pred_target)
    report += "binary f1:\n" + str(s_f1) + lsep
    s_recall = recall_score(ref_target, pred_target)
    s_precision = precision_score(ref_target, pred_target)
    report += "-"*30 + lsep

    modelscore = ModelScore(epoch, s_auc, s_aupr, s_f1, s_precision, s_recall)
    ReaderWriter.write_log(report, outlog)
    return modelscore


def plot_precision_recall_curve(ref_target, prob_poslabel, figname, outdir):
    pr, rec, thresholds = precision_recall_curve(ref_target, prob_poslabel)
    avg_precision = average_precision_score(ref_target, prob_poslabel)
    thresholds[0] = 1
    plt.figure(figsize=(9, 6))
    plt.plot(rec, pr, 'b+', label=f'Average Precision (AP):{avg_precision:.2}')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs. recall curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(outdir, os.path.join('precisionrecall_curve_{}'.format(figname) + ".pdf")))
    plt.close()


def plot_roc_curve(ref_target, prob_poslabel, figname, outdir):
    fpr, tpr, thresholds = roc_curve(ref_target, prob_poslabel)
    thresholds[0] = 1
    plt.figure(figsize=(9, 6))
    plt.plot(fpr, tpr, 'b+', label='TPR vs FPR')
    plt.plot(fpr, thresholds, 'r-', label='thresholds')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(outdir, os.path.join('roc_curve_{}'.format(figname) + ".pdf")))
    plt.close()

def plot_loss(epoch_loss_avgbatch, wrk_dir):
    dsettypes = epoch_loss_avgbatch.keys()
    for dsettype in dsettypes:
        plt.figure(figsize=(9, 6))
        plt.plot(epoch_loss_avgbatch[dsettype], 'r')
        plt.xlabel("number of epochs")
        plt.ylabel("negative loglikelihood cost")
        plt.legend(['epoch batch average loss'])
        plt.savefig(os.path.join(wrk_dir, os.path.join(dsettype + ".pdf")))
        plt.close()

def plot_xy(x, y, xlabel, ylabel, legend, fname, wrk_dir):
    plt.figure(figsize=(9, 6))
    plt.plot(x, y, 'r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend([legend])
    plt.savefig(os.path.join(wrk_dir, os.path.join(fname + ".pdf")))
    plt.close()

def delete_directory(directory):
    if(os.path.isdir(directory)):
        shutil.rmtree(directory)

def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return round(size,2), power_labels[n]+'bytes'

def add_weight_decay_except_attn(model_lst, l2_reg):
    decay, no_decay = [], []
    for m in model_lst:
        for name, param in m.named_parameters():
            if 'queryv' in name:
                no_decay.append(param)
            else: 
                decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_reg}]