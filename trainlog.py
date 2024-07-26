import csv
import os

class Logger:
    def __init__(self, model_name: str, params: dict) -> None:
        self.model_name = model_name

        if not os.path.exists("logs"):
            os.makedirs("logs")
        
        with open(f'logs/{self.model_name}_log.csv', 'a') as file:
            file.write(
                f"num_skip: {params['num_skip']}, learning_rate: {params['learning_rate']}, batch_size: {params['batch_size']}, num_layer: {params['num_layers']}, hidden_dim: {params['hidden_dim']}\n"
            )
            file.write("current_epoch,total_epochs,elapsed_time,loss,val_loss,train_rmse\n")
        
    def write(self, row: str) -> None:
        with open(f'logs/{self.model_name}_log.csv', 'a') as file:
            file.write(row+"\n")

