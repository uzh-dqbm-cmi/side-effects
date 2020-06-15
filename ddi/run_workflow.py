
import os
import itertools
from .utilities import get_device, create_directory, ReaderWriter, perfmetric_report, plot_loss
from .model import NDD_Code
from .model_attn import DDI_Transformer
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

class DDITrfHyperparamConfig:
    def __init__(self, num_attn_heads, num_transformer_units, 
                p_dropout, nonlin_func, mlp_embed_factor, 
                l2_reg, batch_size, num_epochs):
        self.num_attn_heads = num_attn_heads
        self.num_transformer_units = num_transformer_units
        self.p_dropout = p_dropout
        self.nonlin_func = nonlin_func
        self.mlp_embed_factor = mlp_embed_factor
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.num_epochs = num_epochs


    def __repr__(self):
        desc = " num_attn_heads:{}\n num_transformer_units:{}\n p_dropout:{} \n " \
               "nonlin_func:{} \n mlp_embed_factor:{} \n " \
               "l2_reg:{} \n batch_size:{} \n num_epochs: {}".format(self.num_attn_heads,
                                                                     self.num_transformer_units,
                                                                     self.p_dropout, 
                                                                     self.nonlin_func,
                                                                     self.mlp_embed_factor,
                                                                     self.l2_reg, 
                                                                     self.batch_size,
                                                                     self.num_epochs)
        return desc

def generate_models_config(hyperparam_config, similarity_type, model_name, input_dim, fold_num, fdtype, loss_func='nllloss'):

    dataloader_config = {'batch_size': hyperparam_config.batch_size,
                         'num_workers': 0}
    
    config = {'dataloader_config': dataloader_config,
              'model_config': hyperparam_config
             }

    options = {'similarity_type': similarity_type,
               'fold_num': fold_num,
               'input_dim': input_dim,
               'model_name': model_name,
               'num_epochs': hyperparam_config.num_epochs,
               'weight_decay': hyperparam_config.l2_reg,
               'fdtype':fdtype,
               'to_gpu':True,
               'loss_func':loss_func}

    return config, options

def build_custom_config_map(similarity_type, model_name, loss_func='nllloss'):
    if(model_name == 'NDD'):
        hyperparam_config = NDDHyperparamConfig(400,300,0.5,0,200,20)
        input_dim = 1096
    elif(model_name == 'Transformer'):
        hyperparam_config = DDITrfHyperparamConfig(8, 12, 0.3, nn.ReLU(), 2, 0, 200, 20)
        input_dim = 548
    fold_num = -1 
    fdtype = torch.float32
    mconfig, options = generate_models_config(hyperparam_config, similarity_type, model_name, input_dim, fold_num, fdtype, loss_func=loss_func)
    return mconfig, options

def dump_dict_content(dsettype_content_map, dsettypes, desc, wrk_dir):
    for dsettype in dsettypes:
        path = os.path.join(wrk_dir, '{}_{}.pkl'.format(desc, dsettype))
        ReaderWriter.dump_data(dsettype_content_map[dsettype], path)

def hyperparam_model_search(data_partitions, similarity_type, model_name,
                            input_dim, root_dir, run_gpu_map, loss_func='nllloss', 
                            fdtype=torch.float32, num_epochs=25,
                            prob_interval_truemax=0.05, prob_estim=0.95, random_seed=42,
                            per_base=False):
    # fold_num = get_random_run(len(data_partitions), random_seed=random_seed)
    fold_num = 0
    dsettypes = ['train', 'validation']
    hyperparam_options = get_hyperparam_options(prob_interval_truemax, prob_estim, model_name)
    data_partition = data_partitions[run_num]
    for counter, hyperparam_config in enumerate(hyperparam_options):
        mconfig, options = generate_models_config(hyperparam_config, 
                                                  similarity_type,
                                                  model_name,
                                                  input_dim,
                                                  fold_num, fdtype, 
                                                  loss_func=loss_func)
        options['num_epochs'] = num_epochs # override number of ephocs here
        print("Running experiment {} config #{}".format(experiment_desc, counter))
        path = os.path.join(root_dir, 
                            'run_{}'.format(run_num),
                            'config_{}'.format(counter))
        wrk_dir = create_directory(path)

        if options.get('loss_func') == 'bceloss':
            run_ddi(data_partition, dsettypes, mconfig, options, wrk_dir,
                    state_dict_dir=None, to_gpu=True, 
                    gpu_index=fold_gpu_map[fold_num])
        elif options.get('loss_func') == 'nllloss':
             run_ddiTrf(data_partition, dsettypes, mconfig, options, wrk_dir,
                    state_dict_dir=None, to_gpu=True, 
                    gpu_index=fold_gpu_map[fold_num]) 

        print("-"*15)

def run_ddi(data_partition, dsettypes, config, options, wrk_dir,
            state_dict_dir=None, to_gpu=True, gpu_index=0):
    pid = "{}".format(os.getpid())  # process id description
    # get data loader config
    dataloader_config = config['dataloader_config']
    cld = construct_load_dataloaders(data_partition, dsettypes, dataloader_config, wrk_dir)
    # dictionaries by dsettypes
    data_loaders, epoch_loss_avgbatch, score_dict, class_weights, flog_out = cld
    # print(flog_out)
    # print(class_weights)
    device = get_device(to_gpu, gpu_index)  # gpu device
    fdtype = options['fdtype']
    if('train' in class_weights):
        class_weights = class_weights['train'][1].type(fdtype).to(device)  # update class weights to fdtype tensor
    else:
        class_weights = torch.tensor([1]).type(fdtype).to(device)  # weighting all casess equally

    print("class weights", class_weights)
    # binary cross entropy
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction='mean')

    num_epochs = options.get('num_epochs', 50)
    fold_num = options.get('fold_num')

    # parse config dict
    model_config = config['model_config']
    model_name = options['model_name']


    if(model_name == 'NDD'):
        # ddi model
        ddi_model = NDD_Code(D_in=options['input_dim'],
                            H1=model_config.fc1_dim,
                            H2=model_config.fc2_dim,
                            D_out=1,
                            drop=model_config.p_dropout)
    
    # define optimizer and group parameters
    models_param = list(ddi_model.parameters())
    models = [(ddi_model, model_name)]

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

            if(dsettype == 'train'):  # should be only for train
                for m, m_name in models:
                    m.train()
            else:
                for m, m_name in models:
                    m.eval()

            for i_batch, samples_batch in enumerate(data_loader):
                print('batch num:', i_batch)

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
                    y_pred_logit = ddi_model(X_batch)
                    y_pred_prob  = sigmoid(y_pred_logit)
                    y_pred_clss = torch.zeros(y_pred_prob.shape, device=device, dtype=torch.int32)
                    y_pred_clss[y_pred_prob > 0.5] = 1

                    # print('y_pred_logit', y_pred_logit.shape, y_pred_logit.dtype)
                    # print('y_pred_prob', y_pred_prob.shape, y_pred_prob.dtype)
                    # print('y_pred_class', y_pred_clss.shape, y_pred_clss.dtype)
                    # print('y_batch', y_batch.shape, y_batch.dtype)

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

                    # torch.cuda.ipc_collect()
                    # torch.cuda.empty_cache()
            # end of epoch
            # print("+"*35)
            epoch_loss_avgbatch[dsettype].append(epoch_loss/len(data_loader))
            modelscore = perfmetric_report(pred_class, ref_class, prob_scores, epoch+1, flog_out[dsettype])
            prob_scores_arr = np.concatenate(prob_scores, axis=0)

            perf = modelscore.s_aupr
            if(perf > score_dict[dsettype].s_aupr):
                score_dict[dsettype] = modelscore
                if(dsettype == 'validation'):
                    for m, m_name in models:
                        torch.save(m.state_dict(), os.path.join(m_state_dict_dir, '{}.pkl'.format(m_name)))
                if dsettype in {'test', 'validation'}:
                    predictions_df = build_predictions_df(ddi_ids, ref_class, pred_class, prob_scores_arr)
                    predictions_path = os.path.join(wrk_dir, f'predictions_{dsettype}.csv')
                    predictions_df.to_csv(predictions_path)

    if(num_epochs > 1):
        plot_loss(epoch_loss_avgbatch, fig_dir)
    # dump_scores
    dump_dict_content(score_dict, list(score_dict.keys()), 'score', wrk_dir)



def run_ddiTrf(data_partition, dsettypes, config, options, wrk_dir,
            state_dict_dir=None, to_gpu=True, gpu_index=0):
    pid = "{}".format(os.getpid())  # process id description
    # get data loader config
    dataloader_config = config['dataloader_config']
    cld = construct_load_dataloaders(data_partition, dsettypes, dataloader_config, wrk_dir)
    # dictionaries by dsettypes
    data_loaders, epoch_loss_avgbatch, score_dict, class_weights, flog_out = cld
    # print(flog_out)
    # print(class_weights)
    device = get_device(to_gpu, gpu_index)  # gpu device
    fdtype = options['fdtype']

    if('train' in class_weights):
        class_weights = class_weights['train'].type(fdtype).to(device)  # update class weights to fdtype tensor
    else:
        class_weights = torch.tensor([1]*num_classes).type(fdtype).to(device)  # weighting all casess equally

    print("class weights", class_weights)
    loss_func = torch.nn.NLLLoss(weight=class_weights, reduction='mean')  # negative log likelihood loss

    num_epochs = options.get('num_epochs', 50)
    fold_num = options.get('fold_num')

    # parse config dict
    model_config = config['model_config']
    model_name = options['model_name']


    if(model_name == 'Transformer'):
        ddi_model = DDI_Transformer(input_size=options['input_dim'], 
                                    num_attn_heads=model_config.num_attn_heads, 
                                    mlp_embed_factor=model_config.mlp_embed_factor,
                                    nonlin_func=model_config.nonlin_func,
                                    pdropout=model_config.p_dropout, 
                                    num_transformer_units=model_config.num_transformer_units,
                                    pooling_mode=model_config.pooling_mode,
                                    num_classes=2)
    
    # define optimizer and group parameters
    models_param = list(ddi_model.parameters())
    models = [(ddi_model, model_name)]

    if(state_dict_dir):  # load state dictionary of saved models
        for m, m_name in models:
            m.load_state_dict(torch.load(os.path.join(state_dict_dir, '{}.pkl'.format(m_name)), map_location=device))

    # update models fdtype and move to device
    for m, m_name in models:
        m.type(fdtype).to(device)

    if('train' in data_loaders):
        weight_decay = options.get('weight_decay', 1e-4)
        print('weight_decay', weight_decay)
        # see paper Cyclical Learning rates for Training Neural Networks for parameters' choice
        # `https://arxive.org/pdf/1506.01186.pdf`
        # pytorch version >1.1, scheduler should be called after optimizer
        # for cyclical lr scheduler, it should be called after each batch update
        num_iter = len(data_loaders['train'])  # num_train_samples/batch_size
        c_step_size = int(np.ceil(5*num_iter))  # this should be 2-10 times num_iter
        base_lr = 3e-4
        max_lr = 5*base_lr  # 3-5 times base_lr
        optimizer = torch.optim.Adam(models_param, weight_decay=weight_decay, lr=base_lr)
        cyc_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=c_step_size,
                                                        mode='triangular', cycle_momentum=False)

    if ('validation' in data_loaders):
        m_state_dict_dir = create_directory(os.path.join(wrk_dir, 'model_statedict'))

    if(num_epochs > 1):
        fig_dir = create_directory(os.path.join(wrk_dir, 'figures'))

    # dump config dictionaries on disk
    config_dir = create_directory(os.path.join(wrk_dir, 'config'))
    ReaderWriter.dump_data(config, os.path.join(config_dir, 'mconfig.pkl'))
    ReaderWriter.dump_data(options, os.path.join(config_dir, 'exp_options.pkl'))
    # store attention weights for validation and test set
    seqid_fattnw_map = {dsettype: {} for dsettype in data_loaders if dsettype in {'test'}}

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

            if(dsettype == 'train'):  # should be only for train
                for m, m_name in models:
                    m.train()
            else:
                for m, m_name in models:
                    m.eval()

            for i_batch, samples_batch in enumerate(data_loader):
                print('batch num:', i_batch)

                # zero model grad
                if(dsettype == 'train'):
                    optimizer.zero_grad()

                X_batch, y_batch, ids = samples_batch

                X_batch = X_batch.to(device)
                y_batch = y_batch.reshape(-1, 1) # TODO: reshape when preprocessing feature
                y_batch = y_batch.type(torch.int64).to(device)
                # print('ids', ids.shape, ids.dtype)

                with torch.set_grad_enabled(dsettype == 'train'):
                    # print("number of examples in batch:", docs_batch.size(0))
                    num_samples_perbatch = X_batch.size(0)
                    # print("number_samples_per_batch", num_samples_perbatch)
                    logsoftmax_scores, fattn_w_scores = ddi_model(X_batch)

                    if(dsettype in seqid_fattnw_map):
                        seqid_fattnw_map[dsettype].update({sid:fattn_w_scores[c].detach().cpu() for c, sid in enumerate(ids)})
                   
                    __, y_pred_clss = torch.max(logsoftmax_scores, -1)

                    y_pred_prob  = torch.exp(logsoftmax_scores.detach().cpu()).numpy()
                    

                    pred_class.extend(y_pred_clss.view(-1).tolist())
                    ref_class.extend(y_batch.view(-1).tolist())
                    prob_scores.extend(y_pred_prob.tolist())
                    ddi_ids.extend(ids.tolist())

                    loss = loss_func(logsoftmax_scores, y_batch)
                    if(dsettype == 'train'):
                        # print("computing loss")
                        # backward step (i.e. compute gradients)
                        loss.backward()
                        # optimzer step -- update weights
                        optimizer.step()
                        # after each batch step the scheduler
                        cyc_scheduler.step()
                    epoch_loss += loss.item()

                    # torch.cuda.ipc_collect()
                    # torch.cuda.empty_cache()
            # end of epoch
            # print("+"*35)
            epoch_loss_avgbatch[dsettype].append(epoch_loss/len(data_loader))
            modelscore = perfmetric_report(pred_class, ref_class, prob_scores, epoch, flog_out[dsettype])
            prob_scores_arr = np.concatenate(prob_scores, axis=0)

            perf = modelscore.aupr
            best_rec_score = score_dict[dsettype].aupr
            if(perf > best_rec_score):
                score_dict[dsettype] = modelscore
                if(dsettype == 'validation'):
                    for m, m_name in models:
                        torch.save(m.state_dict(), os.path.join(m_state_dict_dir, '{}.pkl'.format(m_name)))
                elif(dsettype == 'test'):
                    # dump attention weights for the test data
                    dump_dict_content(seqid_fattnw_map, ['test'], 'sampleid_fattnw_map', wrk_dir)
                if dsettype in {'test', 'validation'}:
                    predictions_df = build_predictions_df(ddi_ids, ref_class, pred_class, prob_scores_arr)
                    predictions_path = os.path.join(wrk_dir, f'predictions_{dsettype}.csv')
                    predictions_df.to_csv(predictions_path)
                    
    if(num_epochs > 1):
        plot_loss(epoch_loss_avgbatch, fig_dir)
    # dump_scores
    dump_dict_content(score_dict, list(score_dict.keys()), 'score', wrk_dir)

def build_predictions_df(ids, true_class, pred_class, prob_scores):

    prob_scores_dict = {}
    for i in range (prob_scores.shape[-1]):
        prob_scores_dict[f'prob_score_class{i}'] = prob_scores[:, i]

    df_dict = {
        'id': ids,
        'true_class': true_class,
        'pred_class': pred_class
    }
    df_dict.update(prob_scores_dict)
    predictions_df = pd.DataFrame(df_dict)
    predictions_df.set_index('id', inplace=True)
    return predictions_df


def generate_hyperparam_space(model_name):
    if(model_name == 'NDD'):
        fc1_dim = [400]
        fc2_dim = [300]
        l2_reg_vals = [0.0]
        batch_size_vals = [200]
        dropout_vals = [0.5]
        num_epochs_vals = [20]
        opt_lst = [fc1_dim, fc2_dim, dropout_vals, l2_reg_vals, batch_size_vals, num_epochs_vals]
    elif(model_name == 'Transformer'):
        # TODO: add the possible options for transformer model
        embed_dim = [16,32,64,128]
        num_attn_heads = [4,6,8]
        num_transformer_units = [2]
        p_dropout = [0.1, 0.3, 0.5]
        nonlin_func = [nn.ReLU()]
        mlp_embed_factor = [2]
        l2_reg = [1e-4, 1e-3, 1e-2]
        batch_size = [4000]
        num_epochs = [25]
        opt_lst = [embed_dim, num_attn_heads, 
                   num_transformer_units, p_dropout,
                   nonlin_func, mlp_embed_factor,
                   l2_reg, batch_size, num_epochs]

    hyperparam_space = list(itertools.product(*opt_lst))

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


def get_hyperparam_options(prob_interval_truemax, prob_estim, model_name, random_seed=42):
    np.random.seed(random_seed)
    num_trials = compute_numtrials(prob_interval_truemax, prob_estim)
    hyperparam_space = generate_hyperparam_space(model_name)
    if(num_trials > len(hyperparam_space)):
        num_trials = len(hyperparam_space)
    indxs = np.random.choice(len(hyperparam_space), size=num_trials, replace=False)
    if(model_name == 'NDD'):
        hyperconfig_class = NDDHyperparamConfig
    elif(model_name == 'Transformer'):
        hyperconfig_class = DDITrfHyperparamConfig
    # encoder_dim, num_layers, encoder_approach, attn_method, p_dropout, l2_reg, batch_size, num_epochs
    return [hyperconfig_class(*hyperparam_space[indx]) for indx in indxs]


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

def get_best_config_from_hyperparamsearch(hyperparam_search_dir, num_runs=5, num_trials=60, num_metrics=5, metric_indx=4, random_seed=42):
    """Read best models config from all models tested in hyperparamsearch phase
    Args:
        questions: list, of questions [4,5,9,10,11]
        hyperparam_search_dir: string, path root directory where hyperparam models are stored
        num_trials: int, number of tested models (default 60 based on 0.05 interval and 0.95 confidence interval)
                    see :func: `compute_numtrials`
        metric_indx:int, (default 4) using AUC as performance metric to evaluate among the tested models
    """
    # determine best config from hyperparam search
#     run_num = get_random_run(num_runs, random_seed=random_seed)
    fold_num = 0
    run_dir = os.path.join(hyperparam_search_dir, 'fold_{}'.format(run_num))

    scores = np.ones((num_trials, num_metrics))*-1
    exist_flag = False

    for config_num in range(num_trials):
        score_file = os.path.join(run_dir, 'config_{}'.format(config_num), 'score_validation.pkl')
        if(os.path.isfile(score_file)):
            try:
                mscore = ReaderWriter.read_data(score_file)
                print(mscore)
                scores[config_num, 0] = mscore.best_epoch_indx
                scores[config_num, 1] = mscore.binary_f1
                scores[config_num, 2] = mscore.macro_f1
                scores[config_num, 3] = mscore.aupr
                scores[config_num, 4] = mscore.auc
                exist_flag = True
            except Exception as e:
                print(f'exception occured at config_{config_num}')
                continue
        else:
            print("WARNING: hyperparam search dir does not exist: {}".format(score_file))
    if(exist_flag):
        argmax_indx = get_index_argmax(scores, metric_indx)
        mconfig, options = get_saved_config(os.path.join(run_dir, 'config_{}'.format(argmax_indx), 'config'))
        return mconfig, options, argmax_indx, scores
    
    return None

def train_val_run(datatensor_partitions, config_map, train_val_dir, fold_gpu_map, num_epochs=20):
    dsettypes = ['train']
    mconfig, options = config_map
    options['num_epochs'] = num_epochs  # override number of epochs using user specified value
    similarity_type = options['similarity_type']
    for fold_num in datatensor_partitions:
        # update options fold num to the current fold
        options['fold_num'] = fold_num
        data_partition = datatensor_partitions[fold_num]
        # tr_val_dir = create_directory(train_val_dir)
        path = os.path.join(train_val_dir, 'train_val_{}'.format(similarity_type), 'fold_{}'.format(fold_num))
        wrk_dir = create_directory(path)
        # print(wrk_dir)
        # wrk_dir = create_directory('fold_{}'.format(fold_num),create_directory('train_val_{}'.format(similarity_type), train_val_dir))
        if options.get('loss_func') == 'bceloss':
            run_ddi(data_partition, dsettypes, mconfig, options, wrk_dir,
                    state_dict_dir=None, to_gpu=True, 
                    gpu_index=fold_gpu_map[fold_num])
        elif options.get('loss_func') == 'nllloss':
             run_ddiTrf(data_partition, dsettypes, mconfig, options, wrk_dir,
                    state_dict_dir=None, to_gpu=True, 
                    gpu_index=fold_gpu_map[fold_num])  



def test_run(datatensor_partitions, config_map, train_val_dir, test_dir, fold_gpu_map, num_epochs=1):
    dsettypes = ['test']
    mconfig, options = config_map
    options['num_epochs'] = num_epochs  # override number of epochs using user specified value
    similarity_type = options['similarity_type']
    for fold_num in datatensor_partitions:
        # update options fold num to the current fold
        options['fold_num'] = fold_num
        data_partition = datatensor_partitions[fold_num]
        train_dir = create_directory(os.path.join(train_val_dir, 'train_val_{}'.format(similarity_type), 'fold_{}'.format(fold_num)))
        if os.path.exists(train_dir):
            # load state_dict pth
            state_dict_pth = os.path.join(train_dir, 'model_statedict')
            path = os.path.join(test_dir, 'test_{}'.format(similarity_type), 'fold_{}'.format(fold_num))
            test_wrk_dir = create_directory(path)
            if options.get('loss_func') == 'bceloss':
                run_ddi(data_partition, dsettypes, mconfig, options, test_wrk_dir,
                        state_dict_dir=state_dict_pth, to_gpu=True, 
                        gpu_index=fold_gpu_map[fold_num])
            elif options.get('loss_func') == 'nllloss':
                run_ddiTrf(data_partition, dsettypes, mconfig, options, test_wrk_dir,
                        state_dict_dir=state_dict_pth, to_gpu=True, 
                        gpu_index=fold_gpu_map[fold_num])   
        else:
            print('WARNING: train dir not found: {}'.format(path))
