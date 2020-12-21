import flwr as fl
from flwr_latency.fed_avg_latency_strategy import FedAvg

if __name__ == "__main__":

    strategy = FedAvg(fraction_fit=1.0, min_fit_clients=3, min_available_clients=3,)

    fl.server.start_server(config={"num_rounds": 10}, strategy=strategy)
