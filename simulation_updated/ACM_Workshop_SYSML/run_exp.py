# Standard library imports
import os
import sys
import time
import logging
import pickle
from datetime import datetime
import importlib
import random

# Third-party imports
import numpy as np
import torch
import yaml

# Local imports
from server import Server
from client import Client


# Fixing seeds
def fix_seed(seed):

    """
    Fixes random seeds across various libraries for reproducibility.

    Args :
        seed : value of the seed
    Function which allows to fix all the seed and get reproducible results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def setup_client_selection(server, CS_algo, CS_args):
    """
    Sets up the client selection strategy based on the specified algorithm.
    
    Args:
        server: Server instance
        CS_algo (str): Client selection algorithm name
        CS_args (dict): Client selection arguments
        
    Returns:
        tuple: (client_selection_function, updated_CS_args)
    """
    # Get the client selection function from the server instance
    client_selection = getattr(server, f"client_selection_{CS_algo}", None)
    
    if client_selection is None:
        raise ValueError(f"Unknown client selection algorithm: {CS_algo}")
    
    return client_selection, CS_args

def setup_aggregation(server, Agg_algo, Agg_args):
    """
    Sets up the aggregation strategy based on the specified algorithm.
    
    Args:
        server: Server instance
        Agg_algo (str): Aggregation algorithm name
        Agg_args (dict): Aggregation arguments
        
    Returns:
        tuple: (aggregation_function, updated_Agg_args)
    """
    # Get the aggregation function from the server instance
    aggregation = getattr(server, f"aggregate_{Agg_algo}", None)
    
    if aggregation is None:
        raise ValueError(f"Unknown aggregation algorithm: {Agg_algo}")
    
    return aggregation, Agg_args

def main(rounds, seed, client_list, client_selection, CS_args, aggregation, Agg_args, client_train_config, client_test_config):  
    """
    Main function to run the federated learning simulation.
    """

    fix_seed(seed)

    # Initialize cumulative timing metrics
    cumulative_stats = {
        'client_selection_time': 0,
        'actual_training_time': 0,
        'aggregation_time': 0,
        'validation_time': 0
    }
    
    for rnd in range(1, rounds+1):
        CS_args["round"] = rnd
        Agg_args["round"] = rnd
        
        # Time the client selection process
        cs_start_time = time.time()
        selected_cids = client_selection(client_list, CS_args)
        cs_time = time.time() - cs_start_time
        cumulative_stats['client_selection_time'] += cs_time
        
        logger.info(f"SELECTED_CLIENTS:{selected_cids}")
        
        # Track training time across all clients
        total_train_time = 0
        
        for cid in selected_cids:
            # Training time
            train_start_time = time.time()
            client_list[cid].train(round_id=rnd, args=client_train_config)
            cumulative_stats['actual_training_time'] += time.time() - train_start_time
            
        
        # Time the aggregation process
        agg_start_time = time.time()
        global_wts, client_list = aggregation(selected_cids=selected_cids, client_list=client_list, **Agg_args)
        agg_time = time.time() - agg_start_time
        cumulative_stats['aggregation_time'] += agg_time
        
        print(f"Aggregation time: {agg_time:.4f} seconds")
        
        server.model.load_state_dict(global_wts)
        
        # Time the server validation (testing on global test data)
        val_start_time = time.time()
        server.test(round_id=rnd)
        val_time = time.time() - val_start_time
        cumulative_stats['validation_time'] += val_time
        

        print(f"Validation time: {val_time:.4f} seconds")
        
        sim_training_time = max(client_list[cid].time_util[rnd] for cid in selected_cids)
        
        # Log cumulative statistics after each round
        print("\n=== Time Statistics ===")
        print(f"Total Client Selection Time: {cumulative_stats['client_selection_time']:.4f} seconds")
        print(f"Total Actual Training Time: {cumulative_stats['actual_training_time']:.4f} seconds")
        print(f"Simulation Training Round Time: {sim_training_time:.4f} seconds")
        print(f"Total Aggregation Time: {cumulative_stats['aggregation_time']:.4f} seconds")
        print(f"Total Validation Time: {cumulative_stats['validation_time']:.4f} seconds")
        print("===========================\n")
        
        logger.info("=== Time Statistics ===")
        logger.info(f"TOTAL_CLIENT_SELECTION_TIME:{cumulative_stats['client_selection_time']:.4f}")
        logger.info(f"TOTAL_ACTUAL_TRAINING_TIME:{cumulative_stats['actual_training_time']:.4f}")
        logger.info(f"SIM_TRAINING_ROUND_TIME:{sim_training_time:.4f}")
        logger.info(f"TOTAL_AGGREGATION_TIME:{cumulative_stats['aggregation_time']:.4f}")
        logger.info(f"TOTAL_VALIDATION_TIME:{cumulative_stats['validation_time']:.4f}")
        logger.info("===========================")

if __name__ == "__main__":

    """
    This script orchestrates the execution of the federated learning experiment based on a YAML configuration file.

    It performs the following:
    - Parses a user-provided configuration YAML to set up experiment parameters.
    - Dynamically loads the model class specified in the configuration.
    - Initializes the FL clients with pre-defined energy-time Pareto fronts.
    - Loads or creates the initial model weights.
    - Sets up the aggregator (server) and client selection strategy based on the specified algorithm.
    - Executes the federated learning rounds via a call to `main(client_list)`.

    Usage:
        python run_exp.py <path_to_yaml>

    Arguments:
        <path_to_yaml> : Path to the experiment configuration file in YAML format.
    """

    if len(sys.argv) < 2:
        print("Usage: python run_exp.py <path_to_yaml>")
    else:
        # Parsing config file
        config_yaml_path = sys.argv[1]
        exp_config = None
        try:
            with open(config_yaml_path, 'r') as file:
                exp_config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: File not found at {config_yaml_path}")
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
        
        if exp_config is not None:
            CS_algo = exp_config['FL_config']['CS_algo']
            CS_args = exp_config['FL_config']['CS_args']
            Agg_algo = exp_config['FL_config']['Agg_algo']
            Agg_args = exp_config['FL_config']['Agg_args']
            model_id = exp_config['ML_config']['model_id']
            dataset_id = exp_config['ML_config']['dataset_id']
            
            total_rounds = exp_config['FL_config']['total_rounds']
            total_num_clients = exp_config['FL_config']['total_num_clients']
            num_clients_per_round = exp_config['FL_config']['clients_per_round']
            
            client_train_config = exp_config['ML_config']['train_config']
            client_test_config = exp_config['ML_config']['test_config']
            
            torch_device = "cuda" if torch.cuda.is_available() and exp_config['server_config']['use_gpu'] else "cpu"
            seed = exp_config['server_config']['seed']
            
            exp_name = f"{CS_algo}_{model_id}_{dataset_id}"
            
            # Making CS_args and Agg_args
            CS_args = {
                "round": 0,
                "total_rounds": total_rounds,
                "num_clients_per_round": num_clients_per_round,
            }
            if 'CS_args' in exp_config:
                for key, value in exp_config['FL_config']['CS_args'].items():
                    CS_args[key] = value
            
            Agg_args = {}

            if 'Agg_args' in exp_config:
                for key, value in exp_config['FL_config']['Agg_args'].items():
                    Agg_args[key] = value
            
            
            # Loading the Model class dynamically and loading the initial model
            try:
                parent_dir = os.path.dirname(os.path.dirname(exp_config["ML_config"]["model_file_path"]))
                if parent_dir not in sys.path:
                    sys.path.append(parent_dir)

                model_file = os.path.basename(exp_config["ML_config"]["model_file_path"])
                module_name = os.path.splitext(model_file)[0]
                module_path = f"models.{module_name}"

                module = importlib.import_module(module_path)
                model_class = getattr(module, model_id)
            except (ModuleNotFoundError, AttributeError) as e:
                print(f"Error loading {model_id} from {exp_config['ML_config']['model_file_path']}: {e}")
                sys.exit(0)
                
            initial_model_path = os.path.join(exp_config["ML_config"]["initial_model_path"], f"{model_id}.pth")
            if not os.path.exists(initial_model_path):
                model = model_class(cid="Initial Model", args = exp_config["ML_config"]["model_args"])
                torch.save(model, initial_model_path)
                print("New initial model created.")
            
            # Making the save dir
            os.makedirs(os.path.join(exp_config["server_config"]["save_path"], exp_name), exist_ok=True)
            filename = (
                datetime.now().strftime("%d-%m-%Y %H-%M-%S") + ".log"
            )  # Setting the filename from current date and time
            
            logging.basicConfig(
                filename=os.path.join(exp_config["server_config"]["save_path"], exp_name, filename),
                filemode="a",
                format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
                level=logging.DEBUG,
            )
            logger = logging.getLogger("test")
                        
            # Loading client minibatch times
            client_minibatch_time = exp_config['client_config']['minibatch_time']

            # Log the entire configuration
            logger.info("=== EXPERIMENT CONFIGURATION ===")
            for key, value in exp_config.items():
                if isinstance(value, dict):
                    logger.info(f"{key}:")
                    for subkey, subvalue in value.items():
                        logger.info(f"  {subkey}: {subvalue}")
                else:
                    logger.info(f"{key}: {value}")
            logger.info("=============================")
            

            client_list = list()            
            for i in range(total_num_clients):
                client_obj = Client(logger=logger, 
                                    cid=i, 
                                    device=torch_device,
                                    model_class=model_class, 
                                    model_args=exp_config["ML_config"]["model_args"], 
                                    data_path=exp_config["ML_config"]["dataset_dir"], 
                                    dataset_id=dataset_id, 
                                    train_batch_size=exp_config["ML_config"]["train_config"]["train_bs"], 
                                    test_batch_size=exp_config["ML_config"]["test_config"]["test_bs"],
                                    minibatch_time=client_minibatch_time,
                                    )
                client_obj.model.load_state_dict(torch.load(initial_model_path, weights_only=False).state_dict())
                client_list.append(client_obj)
            
            # Making an Aggregator
            server = Server(
                logger=logger,
                device=torch_device,
                model_class=model_class,
                model_args=exp_config["ML_config"]["model_args"],
                data_path=exp_config["ML_config"]["dataset_dir"],
                dataset_id=dataset_id,
                test_batch_size=exp_config["ML_config"]["test_config"]["test_bs"],
                )

            server.model.load_state_dict(torch.load(initial_model_path, weights_only=False).state_dict())
            
                               
            # Setting correct CS function
            
            client_selection, CS_args = setup_client_selection(server, CS_algo, CS_args)
            aggregation, Agg_args = setup_aggregation(server, Agg_algo, Agg_args)

            main(total_rounds, seed, client_list, client_selection, CS_args, aggregation, Agg_args, client_train_config, client_test_config)
