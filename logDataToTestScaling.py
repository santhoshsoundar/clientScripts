import os
import random
import multiprocessing

from verta import Client
from verta.utils import ModelAPI

os.environ['VERTA_EMAIL'] = 'santhosh@verta.ai'
os.environ['VERTA_DEV_KEY'] = '3e078522-e479-4cd2-b78c-04ffcacae3f4'

HOST = "dev.verta.ai"
EXPERIMENT_NAME = "Scaling"

client = Client(HOST)
proj = client.set_project('Scaling Test 100 jobs of 500k models')
expt = client.set_experiment(EXPERIMENT_NAME)

# Hyperparam random choice of values 
c_list = [0.0001, 0.0002, 0.0004]
solver_list=['lgfgs', 'grad']
max_iter_list = [7, 15, 28]

# results into 30 metric or hyp keys
paramKeyLimit = 10

def getMetrics(key_limit):
    metric_obj = {}
    for i in range(key_limit):
        metric_obj['val_acc' + str(i)] = random.uniform(0.5, 0.9)
        metric_obj['loss' + str(i)] = random.uniform(0.6, 0.8)
        metric_obj['acc' + str(i)] = random.uniform(0.6, 0.8)
    return metric_obj

def getHyperaprams(key_limit):
    hyperparam_obj = {}
    for i in range(key_limit):
        hyperparam_obj['C' + str(i)] = random.choice(c_list)
        hyperparam_obj['solver' + str(i)] = random.choice(solver_list)
        hyperparam_obj['max_iter' + str(i)] = random.choice(max_iter_list)
    return hyperparam_obj

def worker():
    """worker function"""
    # range * jobs = expRuns
    for i in range(5000):
        run = client.set_experiment_run()
        run.log_metrics(getMetrics(paramKeyLimit))
        run.log_hyperparameters(getHyperaprams(paramKeyLimit))
    return

if __name__ == '__main__':
    for i in range(100):
        p = multiprocessing.Process(target=worker)
        p.start()
