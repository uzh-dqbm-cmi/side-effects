
import os
import itertools
from .utilities import get_device, create_directory, ReaderWriter, perfmetric_report, plot_loss
from .model import NDD_Code
from .dataset import construct_load_dataloaders
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.multiprocessing as mp


class NDDHyperparamConfig:
    def __init__(self, fc1_dim, fc2_dim, p_dropout, l2_reg, batch_size, num_epochs):
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.p_dropout = p_dropout
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def __repr__(self):
        desc = " fc1_dim:{}\n fc2_dim:{}\n p_dropout:{} \n " \
               "l2_reg:{} \n batch_size:{} \n num_epochs: {}".format(self.fc1_dim,
                                                                     self.fc2_dim,
                                                                     self.p_dropout, 
                                                                     self.l2_reg, 
                                                                     self.batch_size,
                                                                     self.num_epochs)
        return desc


def generate_models_config(hyperparam_config, similarity_type, fold_num, fdtype):


    # currently generic_config is shared across all models
    # leaving it as placeholder such that custom generic configs could be passed :)


    ndd_config = {'input_dim':1096,
                  'fc1_dim':hyperparam_config.fc1_dim,
                  'fc2_dim':hyperparam_config.fc2_dim,
                  'pdropout':hyperparam_config.p_dropout,
                  'to_gpu':True,
                  }
    generic_config = {'fdtype':fdtype}
    dataloader_config = {'batch_size': hyperparam_config.batch_size,
                         'num_workers': 0}
    config = {'dataloader_config': dataloader_config,
              'ndd_config': ndd_config,
              'generic_config': generic_config
             }

    options = {'similarity_type': similarity_type,
               'fold_num': fold_num,
               'num_epochs': hyperparam_config.num_epochs,
               'weight_decay': hyperparam_config.l2_reg}

    return config, options

def build_config_map(similarity_type):
    hyperparam_config = NDDHyperparamConfig(400,300,0.5,0,200,20)
    fold_num = -1 
    mconfig, options = generate_models_config(hyperparam_config, similarity_type, fold_num, torch.float32)
    return mconfig, options

def dump_dict_content(dsettype_content_map, dsettypes, desc, wrk_dir):
    for dsettype in dsettypes:
        path = os.path.join(wrk_dir, '{}_{}.pkl'.format(desc, dsettype))
        ReaderWriter.dump_data(dsettype_content_map[dsettype], path)


def run_ddi(data_partition, dsettypes, config, options, wrk_dir,
            state_dict_dir=None, to_gpu=True, gpu_index=0):
    pid = "{}".format(os.getpid())  # process id description
    # get data loader config
    dataloader_config = config['dataloader_config']
    cld = construct_load_dataloaders(data_partition, dsettypes, dataloader_config, wrk_dir)
    # dictionaries by dsettypes
    data_loaders, epoch_loss_avgbatch, epoch_loss_avgsamples, score_dict, class_weights, flog_out = cld
    # print(class_weights)
    device = get_device(to_gpu, gpu_index)  # gpu device
    generic_config = config['generic_config']
    fdtype = generic_config['fdtype']
    if('train' in class_weights):
        class_weights = class_weights['train'][1].type(fdtype).to(device)  # update class weights to fdtype tensor
    else:
        class_weights = torch.tensor([1]).type(fdtype).to(device)  # weighting all casess equally

    print("class weights", class_weights)
    # loss_func = torch.nn.NLLLoss(weight=class_weights, reduction='mean')  # negative log likelihood loss
    # binary cross entropy
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction='mean')

    num_epochs = options.get('num_epochs', 50)
    fold_num = options.get('fold_num')

    # parse config dict
    ndd_config = config['ndd_config']

    # ddi model
    ndd_model = NDD_Code(D_in=ndd_config['input_dim'],
                         H1=ndd_config['fc1_dim'],
                         H2=ndd_config['fc2_dim'],
                         D_out=1,
                         drop=ndd_config['pdropout'])
    

    # define optimizer and group parameters
    models_param = list(ndd_model.parameters())
    models = [(ndd_model, 'ndd_code')]

    if(state_dict_dir):  # load state dictionary of saved models
        num_train_epochs = 20
        for m, m_name in models: # TODO: update this as it should read best model achieved on validation set
            m.load_state_dict(torch.load(os.path.join(state_dict_dir, '{}_{}.pkl'.format(m_name, num_train_epochs)), map_location=device))

    # update models fdtype and move to device
    for m, m_name in models:
        m.type(fdtype).to(device)

    if('train' in data_loaders):
        weight_decay = options.get('weight_decay', 1e-3)
        optimizer = torch.optim.Adam(models_param, weight_decay=weight_decay, lr=1e-3)
        # see paper Cyclical Learning rates for Training Neural Networks for parameters' choice
        # `https://arxive.org/pdf/1506.01186.pdf`
        # pytorch version >1.1, scheduler should be called after optimizer
        # for cyclical lr scheduler, it should be called after each batch update
        num_iter = len(data_loaders['train'])  # num_train_samples/batch_size
        c_step_size = int(np.ceil(5*num_iter))  # this should be 2-10 times num_iter
        base_lr = 3e-4
        max_lr = 5*base_lr  # 3-5 times base_lr
        cyc_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=c_step_size,
                                                          mode='triangular', cycle_momentum=False)

    # store sentences' attention weights

    # if ('validation' in data_loaders):
    m_state_dict_dir = create_directory(os.path.join(wrk_dir, 'model_statedict'))

    if(num_epochs > 1):
        fig_dir = create_directory(os.path.join(wrk_dir, 'figures'))

    # dump config dictionaries on disk
    config_dir = create_directory(os.path.join(wrk_dir, 'config'))
    ReaderWriter.dump_data(config, os.path.join(config_dir, 'mconfig.pkl'))
    ReaderWriter.dump_data(options, os.path.join(config_dir, 'exp_options.pkl'))
    sigmoid = torch.nn.Sigmoid()
    for epoch in range(num_epochs):
        # print("-"*35)
        for dsettype in dsettypes:
            print("device: {} | similarity_type: {} | fold_num: {} | epoch: {} | dsettype: {} | pid: {}"
                  "".format(device, options.get('similarity_type'), fold_num, epoch, dsettype, pid))
            pred_class = []
            ref_class = []
            prob_scores = []
            ddi_ids = []
            data_loader = data_loaders[dsettype]
            # total_num_samples = len(data_loader.dataset)
            epoch_loss = 0.
            epoch_loss_deavrg = 0.

            if(dsettype == 'train'):  # should be only for train
                for m, m_name in models:
                    m.train()
            else:
                for m, m_name in models:
                    m.eval()

            for i_batch, samples_batch in enumerate(data_loader):
                # print('batch num:', i_batch)

                # zero model grad
                if(dsettype == 'train'):
                    optimizer.zero_grad()

                X_batch, y_batch, ids = samples_batch

                X_batch = X_batch.to(device)
                y_batch = y_batch.reshape(-1, 1) # TODO: reshape when preprocessing feature
                y_batch = y_batch.to(device)
                # print('ids', ids.shape, ids.dtype)

                with torch.set_grad_enabled(dsettype == 'train'):
                    # print("number of examples in batch:", docs_batch.size(0))
                    num_samples_perbatch = X_batch.size(0)
                    # print("number_samples_per_batch", num_samples_perbatch)
                    y_pred_logit = ndd_model(X_batch)
                    y_pred_prob  = sigmoid(y_pred_logit)
                    y_pred_clss = torch.zeros(y_pred_prob.shape, device=device, dtype=torch.int32)
                    y_pred_clss[y_pred_prob > 0.5] = 1

                    # print('y_pred_logit', y_pred_logit.shape, y_pred_logit.dtype)
                    # print('y_pred_prob', y_pred_prob.shape, y_pred_prob.dtype)
                    # print('y_pred_class', y_pred_clss.shape, y_pred_clss.dtype)
                    # print('y_batch', y_batch.shape, y_batch.dtype)

                    if(dsettype == 'test'):
                        pred_class.extend(y_pred_clss.view(-1).tolist())
                        ref_class.extend(y_batch.view(-1).tolist())
                        prob_scores.extend(y_pred_prob.view(-1).tolist())
                        ddi_ids.extend(ids.tolist())

                    loss = loss_func(y_pred_logit, y_batch)
                    if(dsettype == 'train'):
                        # print("computing loss")
                        # backward step (i.e. compute gradients)
                        loss.backward()
                        # optimzer step -- update weights
                        optimizer.step()
                        # after each batch step the scheduler
                        cyc_scheduler.step()
                    epoch_loss += loss.item()
                    # deaverage the loss to deal with last batch with unequal size
                    epoch_loss_deavrg += loss.item() * num_samples_perbatch

                    # torch.cuda.ipc_collect()
                    # torch.cuda.empty_cache()
            # end of epoch
            # print("+"*35)
            epoch_loss_avgbatch[dsettype].append(epoch_loss/len(data_loader))
            epoch_loss_avgsamples[dsettype].append(epoch_loss_deavrg/len(data_loader.dataset))

            modelscore = perfmetric_report(pred_class, ref_class, prob_scores, epoch+1, flog_out[dsettype])
            perf = modelscore.s_auc
            if(perf > score_dict[dsettype].s_auc):
                score_dict[dsettype] = modelscore
            for m, m_name in models:
                torch.save(m.state_dict(), os.path.join(m_state_dict_dir, '{}_{}.pkl'.format(m_name, (epoch+1))))

    if(num_epochs > 1):
        plot_loss(epoch_loss_avgbatch, epoch_loss_avgsamples, fig_dir)

    # dump_scores
    dump_dict_content(score_dict, list(score_dict.keys()), 'score', wrk_dir)
    # this will run once
    if(dsettype == 'test'):
        # save predictions
        predictions_df = build_predictions_df(ddi_ids, ref_class, pred_class, prob_scores)
        predictions_path = os.path.join(wrk_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path)

    # return ref_class, pred_class, prob_scores

def build_predictions_df(ids, true_class, pred_class, prob_scores):
    df_dict = {
        'id': ids,
        'true_class': true_class,
        'pred_class': pred_class,
        'prob_score_class1': prob_scores,
    }
    predictions_df = pd.DataFrame(df_dict)
    predictions_df.set_index('id', inplace=True)
    return predictions_df


def generate_hyperparam_space():
    fc1_dim = [400]
    fc2_dim = [300]
    l2_reg_vals = [0.0]
    batch_size_vals = [200]
    dropout_vals = [0.5]
    num_epochs_vals = [20]
    hyperparam_space = list(itertools.product(*[fc1_dim,  fc2_dim,
                                                dropout_vals, 
                                                l2_reg_vals, 
                                                batch_size_vals,
                                                num_epochs_vals]))
    return hyperparam_space

def compute_numtrials(prob_interval_truemax, prob_estim):
    """ computes number of trials needed for random hyperparameter search
        see `algorithms for hyperparameter optimization paper
        <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__
        Args:
            prob_interval_truemax: float, probability interval of the true optimal hyperparam,
                i.e. within 5% expressed as .05
            prob_estim: float, probability/confidence level, i.e. 95% expressed as .95
    """
    n = np.log(1-prob_estim)/np.log(1-prob_interval_truemax)
    return(int(np.ceil(n))+1)


def get_hyperparam_options(prob_interval_truemax, prob_estim, random_seed=42):
    np.random.seed(random_seed)
    num_trials = compute_numtrials(prob_interval_truemax, prob_estim)
    hyperparam_space = generate_hyperparam_space()
    if(num_trials > len(hyperparam_space)):
        num_trials = len(hyperparam_space)
    indxs = np.random.choice(len(hyperparam_space), size=num_trials, replace=False)
    # encoder_dim, num_layers, encoder_approach, attn_method, p_dropout, l2_reg, batch_size, num_epochs
    return [NDDHyperparamConfig(*hyperparam_space[indx]) for indx in indxs]


def get_random_simtype_fold_per_hyperparam_exp(similarity_types, random_seed=42):
    """Get for each similarity type the fold number to use for identifying optimal hyperparams
    """
    np.random.seed(random_seed)
    simtype_fold = {}
    for sim_type in similarity_types:
        simtype_fold[sim_type] = np.random.randint(5)
    return simtype_fold


def get_saved_config(config_dir):
    options = ReaderWriter.read_data(os.path.join(config_dir, 'exp_options.pkl'))
    mconfig = ReaderWriter.read_data(os.path.join(config_dir, 'mconfig.pkl'))
    return mconfig, options


def get_index_argmax(score_matrix, target_indx):
    argmax_indx = np.argmax(score_matrix, axis=0)[target_indx]
    return argmax_indx


def train_val_run(datatensor_partitions, config_map, train_val_dir, fold_gpu_map, num_epochs=20):
    dsettypes = ['train']
    mconfig, options = config_map
    options['num_epochs'] = num_epochs  # override number of epochs using user specified value
    similarity_type = options['similarity_type']
    for fold_num in datatensor_partitions:
        # update options fold num to the current fold
        options['fold_num'] = fold_num
        data_partition = datatensor_partitions[fold_num]
        path = os.path.join(train_val_dir, 'train_val_{}'.format(similarity_type), 'fold_{}'.format(fold_num))
        wrk_dir = create_directory(path)
        run_ddi(data_partition, dsettypes, mconfig, options, wrk_dir,
                state_dict_dir=None, to_gpu=True, gpu_index=fold_gpu_map[fold_num])


def test_run(datatensor_partitions, config_map, train_val_dir, test_dir, fold_gpu_map, num_epochs=1):
    dsettypes = ['test']
    mconfig, options = config_map
    options['num_epochs'] = num_epochs  # override number of epochs using user specified value
    similarity_type = options['similarity_type']
    for fold_num in datatensor_partitions:
        # update options fold num to the current fold
        options['fold_num'] = fold_num
        data_partition = datatensor_partitions[fold_num]
        path = os.path.join(train_val_dir, 'train_val_{}'.format(similarity_type), 'fold_{}'.format(fold_num))
        if os.path.exists(path):
            train_dir = create_directory(path)
            # load state_dict pth
            state_dict_pth = os.path.join(train_dir, 'model_statedict')
            path = os.path.join(test_dir, 'test_{}'.format(similarity_type), 'fold_{}'.format(fold_num))
            test_wrk_dir = create_directory(path)
            run_ddi(data_partition, dsettypes, mconfig, options, test_wrk_dir,
                    state_dict_dir=state_dict_pth, to_gpu=True, 
                    gpu_index=fold_gpu_map[fold_num])
        else:
            print('WARNING: test dir not found: {}'.format(path))

