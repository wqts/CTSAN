import wandb
import os
import multiprocessing
import collections

from lib.utils import git_commitID_get

import numpy as np

if os.path.exists("/root/autodl-tmp/"):
    wandb_log_dir = "/root/autodl-tmp/"
else:
    wandb_log_dir = "./"

os.environ["WANDB_DIR"] = os.path.abspath(wandb_log_dir)

Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("num", "dir", "project", "sweep_id", "sweep_run_name", "config")
)

def main(config, main_worker, Debug=True):
    config["git_id"] = git_commitID_get()
    if Debug == True:
        folds = 1
        project = "Debug"
    else:
        if config["dataset"] in ["SEED", "SEED-IV"]:
            folds = 15
        else:
            folds = 10
        project = "Release"
    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    sweep_q = multiprocessing.Queue()
    workers = []
    for num in range(folds):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=main_worker, kwargs=dict(sweep_q=sweep_q, worker_q=q)
        )
        p.start()
        workers.append(Worker(queue=q, process=p))

    sweep_run = wandb.init(dir=wandb_log_dir, project="Sweep", config=config)
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    sweep_run_name = sweep_run.id
    sweep_run.name = sweep_run_name
    sweep_run.save()

    metrics = []
    for num in range(folds):
        worker = workers[num]
        # start worker
        worker.queue.put(
            WorkerInitData(
                num=num,
                dir=wandb_log_dir,
                project=project,
                sweep_id=sweep_id,
                sweep_run_name=sweep_run_name,
                config=dict(sweep_run.config),
            )
        )

    for num in range(folds):
        worker = workers[num]
        # get metric from worker
        result = sweep_q.get()
        # wait for worker to finish
        worker.process.join()
        # log metric to sweep_run
        metrics.append(result.best_accuracy)
    metrics = np.array(metrics)
    sweep_run.log(dict(accuracy_mean=metrics.mean(), accuracy_std=metrics.std()))
    wandb.join()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)