import collections
import wandb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from lib.load_data import load_data
from lib.model import Model
from lib.pipeline import main
from lib.utils import test, reset_wandb_env, get_batch_indices, sharpen

from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss

WorkerDoneData = collections.namedtuple("WorkerDoneData", ("best_accuracy"))

def train_epoch(config, dset_dict, model_de, model_psd, domain_advs_de, domain_advs_psd, optimizer_de, optimizer_psd, cur_epoch):
    loss_fn = nn.CrossEntropyLoss()
    model_de.train()
    model_psd.train()
    for i in range(config.it_per_epoch):
        lb_indices = get_batch_indices(dset_dict['lb_dset'], config.batch_size)
        ulb_indices = get_batch_indices(dset_dict['ulb_dset'], config.batch_size)
        tar_indices = get_batch_indices(dset_dict['tar_dset'], config.batch_size)
        de_lb = torch.index_select(dset_dict['lb_dset']['de'], 0, lb_indices)
        psd_lb = torch.index_select(dset_dict['lb_dset']['psd'], 0, lb_indices)
        y_lb = torch.index_select(dset_dict['lb_dset']['label'], 0, lb_indices)
        de_ulb = torch.index_select(dset_dict['ulb_dset']['de'], 0, ulb_indices)
        psd_ulb = torch.index_select(dset_dict['ulb_dset']['psd'], 0, ulb_indices)
        de_tar = torch.index_select(dset_dict['tar_dset']['de'], 0, tar_indices)
        psd_tar = torch.index_select(dset_dict['tar_dset']['psd'], 0, tar_indices)
        # predict
        pred_de_lb = model_de(de_lb)
        pred_psd_lb = model_psd(psd_lb)
        logits_de_lb = pred_de_lb["logits"]
        logits_psd_lb = pred_psd_lb["logits"]
        feature_de_lb = pred_de_lb["feature"]
        feature_psd_lb = pred_psd_lb["feature"]
        # Compute prediction error
        cls_loss_de_lb = loss_fn(logits_de_lb, y_lb)
        cls_loss_psd_lb = loss_fn(logits_psd_lb, y_lb)

        pred_de_ulb = model_de(de_ulb)
        pred_psd_ulb = model_psd(psd_ulb)
        logits_de_ulb = pred_de_ulb["logits"]
        logits_psd_ulb = pred_psd_ulb["logits"]
        feature_de_ulb = pred_de_ulb["feature"]
        feature_psd_ulb = pred_psd_ulb["feature"]
        pred_de_tar = model_de(de_tar)
        pred_psd_tar = model_psd(psd_tar)
        logits_de_tar = pred_de_tar["logits"]
        logits_psd_tar = pred_psd_tar["logits"]
        feature_de_tar = pred_de_tar["feature"]
        feature_psd_tar = pred_psd_tar["feature"]
        # generate pseudo-label
        pseudo_labels_de_ulb = sharpen(torch.softmax(logits_de_ulb.detach(), dim=1), config.t)
        pseudo_labels_psd_ulb = sharpen(torch.softmax(logits_psd_ulb.detach(), dim=1), config.t)
        pseudo_labels_de_tar = sharpen(torch.softmax(logits_de_tar.detach(), dim=1), config.t)
        pseudo_labels_psd_tar = sharpen(torch.softmax(logits_psd_tar.detach(), dim=1), config.t)

        transfer_loss_de = 0
        for j, domain_adv in enumerate(domain_advs_de):
            sub_domain_transfer_loss_de = domain_adv(torch.cat((feature_de_lb, feature_de_ulb)), torch.cat((feature_de_tar, feature_de_tar)))
            weight = torch.cat((torch.eye(config.n_class).cuda()[y_lb], pseudo_labels_de_ulb, pseudo_labels_de_tar, pseudo_labels_de_tar))[:, j]
            transfer_loss_de += (weight * torch.squeeze(sub_domain_transfer_loss_de, dim=1)).mean()

        transfer_loss_psd = 0
        for j, domain_adv in enumerate(domain_advs_psd):
            sub_domain_transfer_loss_psd = domain_adv(torch.cat((feature_psd_lb, feature_psd_ulb)), torch.cat((feature_psd_tar, feature_psd_tar)))
            weight = torch.cat((torch.eye(config.n_class).cuda()[y_lb], pseudo_labels_psd_ulb, pseudo_labels_psd_tar, pseudo_labels_psd_tar))[:, j]
            transfer_loss_psd += (weight * torch.squeeze(sub_domain_transfer_loss_psd, dim=1)).mean() / config.n_class

        loss_de = cls_loss_de_lb + config.transfer_weight * transfer_loss_de
        loss_psd = cls_loss_psd_lb + config.transfer_weight * transfer_loss_psd

        # Backpropagation
        loss_de.backward()
        optimizer_de.step()
        optimizer_de.zero_grad()

        loss_psd.backward()
        optimizer_psd.step()
        optimizer_psd.zero_grad()

def main_worker(sweep_q, worker_q):
    reset_wandb_env()
    worker_data = worker_q.get()
    run_name = "{}-{}".format(worker_data.sweep_run_name, worker_data.num)
    run = wandb.init(
        dir=worker_data.dir,
        project=worker_data.project,
        group=worker_data.sweep_id,
        job_type=worker_data.sweep_run_name,
        name=run_name,
        config=worker_data.config,
    )
    config = wandb.config
    fold = worker_data.num + 1

    dset_dict = load_data(config, fold)

    model_de = Model(config).cuda()
    domain_advs_de = []
    params_de = list(model_de.parameters())
    for i in range(config.n_class):
        domain_discri = DomainDiscriminator(in_feature=64, hidden_size=64).cuda()
        domain_advs_de.append(DomainAdversarialLoss(domain_discri, 'none').cuda())
        params_de += list(domain_discri.parameters())
    optimizer_de = Adam(params=params_de, lr=config.lr)

    model_psd = Model(config).cuda()
    domain_advs_psd = []
    params_psd = list(model_psd.parameters())
    for i in range(config.n_class):
        domain_discri = DomainDiscriminator(in_feature=64, hidden_size=64).cuda()
        domain_advs_psd.append(DomainAdversarialLoss(domain_discri, 'none').cuda())
        params_psd += list(domain_discri.parameters())
    optimizer_psd = Adam(params=params_psd, lr=config.lr)
    de_best_accuracy = 0
    psd_best_accuracy = 0

    if fold == 1:
        print("This is the train process for subject1, and only subject1 is printed to simplify the output on the console.")

    for i in range(config.epochs):
        train_epoch(config, dset_dict, model_de, model_psd, domain_advs_de, domain_advs_psd, optimizer_de, optimizer_psd, i)
        de_lb_loss, de_lb_accuracy = test(config, dset_dict['lb_dset'], model_de, feature='de')
        de_ulb_loss, de_ulb_accuracy = test(config, dset_dict['ulb_dset'], model_de, feature='de')
        de_test_loss, de_test_accuracy = test(config, dset_dict['tar_dset'], model_de, feature='de')
        psd_lb_loss, psd_lb_accuracy = test(config, dset_dict['lb_dset'], model_psd, feature='psd')
        psd_ulb_loss, psd_ulb_accuracy = test(config, dset_dict['ulb_dset'], model_psd, feature='psd')
        psd_test_loss, psd_test_accuracy = test(config, dset_dict['tar_dset'], model_psd, feature='psd')
        de_best_accuracy = max(de_best_accuracy, de_test_accuracy)
        psd_best_accuracy = max(psd_best_accuracy, psd_test_accuracy)
        best_accuracy = max(de_best_accuracy, psd_best_accuracy)
        if fold == 1:
            print(f"epoch{i + 1}: de_lb_loss={de_lb_loss:.4f}, de_lb_accuracy={de_lb_accuracy:.4f}, de_ulb_loss={de_ulb_loss:.4f}, de_ulb_accuracy={de_ulb_accuracy:.4f}, de_test_loss={de_test_loss:.4f}, de_test_accuracy={de_test_accuracy:.4f}, de_best_accuracy={de_best_accuracy:.4f}")
            print(f"epoch{i + 1}: psd_lb_loss={psd_lb_loss:.4f}, psd_lb_accuracy={psd_lb_accuracy:.4f}, psd_ulb_loss={psd_ulb_loss:.4f}, psd_ulb_accuracy={psd_ulb_accuracy:.4f}, psd_test_loss={psd_test_loss:.4f}, psd_test_accuracy={psd_test_accuracy:.4f}, psd_best_accuracy={psd_best_accuracy:.4f}")
        run.log({'de/lb_loss':de_lb_loss, 'de/lb_accuracy':de_lb_accuracy, 'de/ulb_loss':de_ulb_loss, 'de/ulb_accuracy':de_ulb_accuracy, 'de/test_loss':de_test_loss, 'de/test_accuracy':de_test_accuracy, 'de/best_accuracy':de_best_accuracy, 'psd/lb_loss':psd_lb_loss, 'psd/lb_accuracy':psd_lb_accuracy, 'psd/ulb_loss':psd_ulb_loss, 'psd/ulb_accuracy':psd_ulb_accuracy, 'psd/test_loss':psd_test_loss, 'psd/test_accuracy':psd_test_accuracy, 'psd/best_accuracy':psd_best_accuracy, 'best_accuracy': best_accuracy})
    wandb.join()
    sweep_q.put(WorkerDoneData(best_accuracy=best_accuracy))
    

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    default_config_SEED={
        # data
        "dataset": "SEED",
        "num_subject": 15,
        "n_class": 3,
        "n_labeled_subject": 13,
        "in_feature": 310,
        
        # alg
        "alg": "ctsan",
        "batch_size": 64,
        "epochs": 200,
        "it_per_epoch": 50,
        "optimizer": "Adam",
        "lr": 1e-3,

        # specific
        "t": 0.9,
        "transfer_weight": 1,
    }
    default_config_SEED_IV={
        # data
        "dataset": "SEED-IV",
        "num_subject": 15,
        "n_class": 4,
        "n_labeled_subject": 13,
        "in_feature": 310,

        # alg
        "alg": "ctsan",
        "batch_size": 64,
        "epochs": 200,
        "it_per_epoch": 50,
        "optimizer": "Adam",
        "lr": 1e-3,

        # specific
        "t": 0.9,
        "transfer_weight": 1,
    }
    Debug = True
    # Debug = False
    main(config=default_config_SEED_IV, main_worker=main_worker, Debug=Debug)