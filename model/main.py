import os
import json
import numpy as np
import torch as pt
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.logger import Logger
from src.scoring import bc_scoring, bc_score_names, nanmean
from config import config_data, config_model, config_runtime
from data_handler import collate_batch_data
from src.dataset import StructuresDataset
from model import Model


def setup_dataloader(config_data, sids_selection_filepath):

    # create dataset
    dataset = StructuresDataset(sids_selection_filepath['wt_pdbs'], sids_selection_filepath['skempi'], max_chain_size=config_data['max_chain_size'])
    
    val_size = int(0.1764 * len(dataset))
    train_size = len(dataset) - val_size
    
    train_data, val_data = random_split(dataset, 
                                        [train_size, val_size])
                                        #generator=torch.Generator().manual_seed(42))

    # define data loader
    train_dataloader = pt.utils.data.DataLoader(train_data,
                                          batch_size=config_runtime['batch_size'],
                                          shuffle=True, #num_workers=8,
                                          collate_fn=collate_batch_data,
                                          pin_memory=True)
    
    val_dataloader = pt.utils.data.DataLoader(val_data,
                                          batch_size=config_runtime['batch_size'],
                                          shuffle=True, #num_workers=8,
                                          collate_fn=collate_batch_data,
                                          pin_memory=True)
                                          #prefetch_factor=2)

    return train_dataloader, val_dataloader


def eval_step(model, device, batch_data, criterion, pos_ratios, pos_weight_factor, global_step, dyn_pos_ratio = True):
    # unpack data
    X, ids_topk, q, M, mut, batch_M, y = [data.to(device) for data in batch_data]

    # run model
    z = model.forward(X, ids_topk, q, M, mut, batch_M)

    # compute weighted loss
    if(dyn_pos_ratio):
        mean_y = pt.mean(y, dim=0)
        pos_ratios += (pt.tensor([mean_y, 1-mean_y]).to(device).detach() - pos_ratios) / (1.0 + np.sqrt(global_step))
        criterion.pos_weight = pos_weight_factor * (1.0 - pos_ratios) / (pos_ratios + 1e-6)
    z = pt.unsqueeze(z, dim=-1)
    y = pt.unsqueeze(y, dim=-1)
    z = pt.cat([z, -z], dim=-1)
    y = pt.cat([y, 1-y], dim=-1)
    losses = criterion(z, y)

    return losses, y.detach(), pt.sigmoid(z).detach()


def scoring(eval_results, device=pt.device('cpu')):
    # compute sum losses and scores for each entry
    sum_losses, scores = [], []
    ys = []
    ps = []
    outs = []

    for losses, y, p in eval_results:
        sum_losses.append(pt.sum(losses, dim=0))
        ys.append(y)
        ps.append(p)
        outs.append(pt.sum(p, dim=0))

    # average scores
    m_losses = pt.mean(pt.stack(sum_losses, dim=0), dim=0).numpy()
    ys = pt.cat(ys)
    ps = pt.cat(ps)
    m_scores = bc_scoring(ys, ps)
    m_p = pt.mean(pt.stack(outs, dim=0), dim=0).numpy()
    print(ys)
    print(ps)

    # pack scores
    scores = {'loss': float(np.sum(m_losses))}

    for i in range(m_losses.shape[0]):
        scores[f'{i}/loss'] = m_losses[i]
        scores[f'{i}/outs'] = m_p[i]
        for j in range(m_scores.shape[0]):
            scores[f'{i}/{bc_score_names[j]}'] = m_scores[j,i]

    return scores


def logging(logger, writer, scores, global_step, pos_ratios, step_type):
    # debug print
    pr_str = ', '.join([f"{r:.4f}" for r in pos_ratios])
    logger.print(f"{step_type}> [{global_step}] loss={scores['loss']:.4f}, pos_ratios=[{pr_str}]")

    # store statistics
    summary_stats = {k:scores[k] for k in scores if not np.isnan(scores[k])}
    summary_stats['global_step'] = int(global_step)
    summary_stats['pos_ratios'] = list(pos_ratios.cpu().numpy())
    summary_stats['step_type'] = step_type
    #logger.store(**summary_stats)

    # detailed information
    for key in scores:
        writer.add_scalar(step_type+'/'+key, scores[key], global_step)

    # debug print
    for c in np.unique([key.split('/')[0] for key in scores if len(key.split('/')) == 2]):
        logger.print(f'[{c}] loss={scores[c+"/loss"]:.3f}, ' + ', '.join([f'{sn}={scores[c+"/"+sn]:.3f}' for sn in bc_score_names]))


def train(config_data, config_model, config_runtime, output_path):
    # create logger
    logger = Logger(output_path, 'train')

    # print configuration
    logger.print(">>> Configuration")
    logger.print(config_data)
    logger.print(config_runtime)

    # define device
    device = pt.device(config_runtime['device'])

    # create model
    model = Model(config_model)

    # debug print
    logger.print(">>> Model")
    logger.print(model)
    logger.print(f"> {sum([int(pt.prod(pt.tensor(p.shape))) for p in model.parameters()])} parameters")

#   TODO: Check and fix this to work
#    # reload model if configured
#    model_filepath = os.path.join(output_path, 'model_ckpt.pt')
#    if os.path.isfile(model_filepath) and config_runtime["reload"]:
#        logger.print("Reloading model from save file")
#        model.load_state_dict(pt.load(model_filepath))
#        # get last global step
#        global_step = json.loads([l for l in open(logger.log_lst_filepath, 'r')][-1])['global_step']
#        # dynamic positive weight
#       if dyn_pos ratio TODO
#        pos_ratios = pt.from_numpy(np.array(json.loads([l for l in open(logger.log_lst_filepath, 'r')][-1])['pos_ratios'])).float().to(device)
#    else:
    # starting global step
    global_step = 0
    
    # dynamic positive weight
    if(config_runtime['dyn_pos_ratio']):
        pos_ratios = 0.5*pt.ones(2, dtype=pt.float).to(device)
    else:
        pos_ratios = pt.tensor(config_runtime['pos_ratios']).to(device)

    # debug print
    logger.print(">>> Loading data")

    # setup dataloaders
    dataloader_train, dataloader_val = setup_dataloader(config_data, config_data['train_selection_filepath'])

    # debug print
    logger.print(f"> training data size: {len(dataloader_train)}")
    logger.print(f"> testing data size: {len(dataloader_val)}")

    # debug print
    logger.print(">>> Starting training")

    # send model to device
    model = model.to(device)

    # define losses functions
    if (config_runtime['dyn_pos_ratio']):
        criterion = pt.nn.BCEWithLogitsLoss(reduction="none")
    else:
        criterion = pt.nn.BCEWithLogitsLoss(reduction="none",
                                            weight=pos_weight_factor * (1.0 - pos_ratios) / (pos_ratios + 1e-6)
                                           )

    # define optimizer
    optimizer = pt.optim.Adam(model.parameters(), lr=config_runtime["learning_rate"])
    #optimizer = pt.optim.Adam([
    #    {'params': model.em.parameters(), 'lr': 1e-4},
    #    {'params': model.sum.parameters(), 'lr': 1e-3},
    #    {'params': model.dm.parameters(), 'lr': 1e-4},
    #    {'params': model.mutm.parameters(), 'lr': 1e-4},
    #    {'params': model.spl.parameters(), 'lr': 1e-4},
    #    {'params': model.aggr.parameters(), 'lr': 1e-4},
    #    {'params': model.out.parameters(), 'lr': 1e-4}
    #])

    # restart timer
    logger.restart_timer()

    # summary writer
    writer = SummaryWriter(os.path.join(output_path, 'tb'))

    # min loss initial value
    min_loss = 1e9

    # TODO: Implement get_largest
    # quick training step on largest data: memory check and pre-allocation
    #batch_data = dataloader_train.dataset.get_largest()
    #optimizer.zero_grad()
    #losses, _, _ = eval_step(model, device, batch_data, criterion, pos_ratios.to(device), config_runtime['pos_weight_factor'], global_step, dyn_pos_ratio=config_runtime['dyn_pos_ratio'])
    #loss = pt.sum(losses)
    #loss.backward()
    #optimizer.step()

    # start training
    for epoch in range(config_runtime['num_epochs']):
        # train mode
        model = model.train()

        # train model
        train_results = []
        for batch_train_data in tqdm(dataloader_train):
            # global step
            global_step += 1

            # forward propagation
            losses, y, p = eval_step(model, device, batch_train_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step, dyn_pos_ratio=config_runtime['dyn_pos_ratio'])

            # backward propagation
            loss = pt.sum(losses)/config_runtime["optimizer_step"]
            loss.backward()


            # gradients in sum layers
            sum_=0
            for parm in model.sum.parameters():
                if(parm.requires_grad):
                    sum_= sum_ + np.abs(parm.grad.data.cpu().numpy()).sum()
            writer.add_scalar("train/grad_sum", sum_, global_step)

            # gradients in the whole model
            sum_=0
            for parm in model.parameters():
                if(parm.requires_grad):
                    sum_= sum_ + np.abs(parm.grad.data.cpu().numpy()).sum()
            writer.add_scalar("train/grad", sum_, global_step)

            # store evaluation results
            train_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])

            # optimization step
            if (global_step+1) % config_runtime["optimizer_step"] == 0:
                #pt.nn.utils.clip_grad_norm_(model.parameters(), 20, norm_type=1)
                optimizer.step()
                optimizer.zero_grad()

            # log step
            if (global_step+1) % config_runtime["log_step"] == 0:
                # process evaluation results

                with pt.no_grad():
                    # scores evaluation results and reset buffer
                    scores = scoring(train_results, device=device)
                    train_results = []

                    # logging
                    logging(logger, writer, scores, global_step, pos_ratios, "train")

                    # save model checkpoint
                    model_filepath = os.path.join(output_path, 'model_ckpt.pt')
                    pt.save(model.state_dict(), model_filepath)

            # evaluation step
            if (global_step+1) % config_runtime["eval_step"] == 0:
                # evaluation mode
                model = model.eval()

                with pt.no_grad():
                    # evaluate model
                    test_results = []
                    for step_te, batch_test_data in enumerate(dataloader_val):
                        # forward propagation
                        losses, y, p = eval_step(model, device, batch_test_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step, dyn_pos_ratio=config_runtime['dyn_pos_ratio'])

                        # store evaluation results
                        test_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])

                        # stop evaluating
                        if step_te >= config_runtime['eval_size']:
                            break

                    # scores evaluation results
                    scores = scoring(test_results, device=device)

                    # logging
                    logging(logger, writer, scores, global_step, pos_ratios, "test")

                    # save model and update min loss
                    if min_loss >= scores['loss']:
                        # update min loss
                        min_loss = scores['loss']
                        # save model
                        model_filepath = os.path.join(output_path, 'model.pt')
                        logger.print("> saving model at {}".format(model_filepath))
                        pt.save(model.state_dict(), model_filepath)

                # back in train mode
                model = model.train()


if __name__ == '__main__':
    # train model
    train(config_data, config_model, config_runtime, '.')
