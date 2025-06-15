import pickle
import numpy as np
import random 
from collections import OrderedDict
import torch
from itertools import combinations
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from collections import deque




class ClientSelection:
    
    def __init__(self):
        pass

    """
    Client Selection Algorithms
    """
    
    def client_selection_random(self, clients, args: dict) -> list:
        return np.random.choice([client.cid for client in clients], args["num_clients_per_round"], replace=False).tolist()

    def client_selection_score_based(self, clients, args: dict) -> list:
        scores = []
        for client in clients:
            if len(client.accuracy_history) >= 2:
                acc_gain = client.accuracy_history[-1] - np.mean(client.accuracy_history[:-1])
            elif len(client.accuracy_history) == 1:
                acc_gain = client.accuracy_history[-1]
            else:
                acc_gain = 0.0

            time_factor = client.minibatch_time if client.minibatch_time > 0 else 1.0

            if hasattr(client, "label_distribution") and client.label_distribution:
                probs = np.array(list(client.label_distribution.values()))
                probs = probs / np.sum(probs)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
            else:
                entropy = 1.0

            score = (acc_gain / time_factor) * entropy * np.log1p(client.num_items)
            scores.append((client.cid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [cid for cid, _ in scores[:args["num_clients_per_round"]]]
        return selected

    
class Aggregation:
    def __init__(self):
        pass
    """
    Aggregation Algorithms
    """
    
    def aggregate_fedavg(self, round, selected_cids, client_list, update_client_models = True):
        
        global_model = OrderedDict()
        client_local_weights = client_list[0].model.to("cpu").state_dict()
        
        for layer in client_local_weights:
            shape = client_local_weights[layer].shape
            global_model[layer] = torch.zeros(shape)

        client_weights = list()
        
        n_k = list()
        for client_id in selected_cids:
            client_weights.append(client_list[client_id].model.to("cpu").state_dict())
            n_k.append(client_list[client_id].num_items)

        n_k = np.array(n_k)
        n_k = n_k / sum(n_k)
        
        for i, weights in enumerate(client_weights):
            for layer in weights.keys():
                # fmt: off
                global_model[layer] += (weights[layer] * n_k[i])
                # fmt: on

        # print("Global Model :: ", global_model["conv1.weight"][0])
        if update_client_models:
            for client in client_list:
                client.model.load_state_dict(global_model)

        return global_model, client_list
    
    
class Server(ClientSelection, Aggregation):
    def __init__(self, logger, device, model_class, model_args, data_path, dataset_id, test_batch_size):
        ClientSelection.__init__(self)
        Aggregation.__init__(self)
        
        self.id = "server"
        self.device = device
        self.logger = logger
        self.model = model_class(self.id, model_args)
        
        # Load normal test data for evaluation
        _, self.test_data = self.model.load_data(logger, data_path, dataset_id, self.id, None, test_batch_size)

        self.test_metrics = dict()  


    def test(self, round_id):
        data = self.test_data
        self.test_metrics[round_id] = self.model.test_model(self.logger, data)

            
            