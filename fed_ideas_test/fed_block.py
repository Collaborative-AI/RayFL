import numpy as np
import ray
import random
import time



@ray.remote
class Model:
    def __init__(self):
        self._model = [0]
# generate 50 data points each epoch
    def load_data(self):
        return

    def train(self):
        # time.sleep(random.uniform(1, 6))
        time.sleep(20)
        self._model[0] +=3
        

    def get_weights(self):
        return self._model
    
    def update_weights(self, weights):
        self._model = weights

    def predict(self, x):
        return
# 

def averge(weights):
    return [np.mean(weights)]

def run_experiment(num_clients, rounds):
    clients = [Model.remote() for _ in range(num_clients)]
    for round in range(rounds):
        # create num_clients
        train = [client.train.remote() for client in clients]
        # get will wait every one ends
        # done_ids, not_done_ids =  ray.wait(train)
        # print("at round", round, "time is", time.time() - start)
        # get models, update model to server
        weights = ray.get([client.get_weights.remote() for client in clients])
        mean_weight = averge(weights=weights)
        [client.update_weights.remote(mean_weight) for client in clients]
        weights = ray.get([client.get_weights.remote() for client in clients])
        print("all weights after update", weights)
        print()

start = time.time()
run_experiment(40, 1)
print("duration =", time.time() - start)
