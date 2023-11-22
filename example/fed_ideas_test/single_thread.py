import numpy as np
import random
import time



# without ray
class Model:
    def __init__(self):
        self._model = [0]
    def load_data(self):
        return

    def train(self):
        time.sleep(20)
        self._model[0] +=3
        

    def get_weights(self):
        return self._model
    
    def update_weights(self, weights):
        self._model = weights

    def predict(self, x):
        return

def averge(weights):
    return [np.mean(weights)]

start = time.time()
def run_experiment(num_clients, rounds):
    clients = [Model() for _ in range(num_clients)]
    for _ in range(rounds):
        for i in range(num_clients):
            clients[i].train()
            print("at round", _, "client", i, "time is", time.time() - start)
        # get models, update model to server
        weights = [client.get_weights() for client in clients]
        mean_weight = averge(weights=weights)
        [client.update_weights(mean_weight) for client in clients]
        weights = [client.get_weights() for client in clients]
        print("all weights after update", weights)
        print()
    

run_experiment(40, 1)
print("fed block total time is", time.time() - start, "seconds")


            

    