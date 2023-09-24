import os
import datetime
import yaml
from globals import config, hyperparameters
import torch
import copy
from View.make_plots import process_and_save_images_as_pdf

class ExperimentLogger:
    def __init__(self):
        # Set the name based on the package and system date and time
        package_name = os.path.basename(os.path.dirname(__file__))  # Assuming file resides directly inside package folder
        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.name = f"{package_name}_{current_time}"

        # Set the directory path
        self.dir = os.path.join(config()["project_dir"], self.name)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # Store copies of hyperparameters and config as YAML files
        with open(os.path.join(self.dir, 'hyperparameters.yml'), 'w') as hyper_file:
            yaml.dump(hyperparameters(), hyper_file)
        with open(os.path.join(self.dir, 'config.yml'), 'w') as config_file:
            con_copy = copy.deepcopy(config())
            con_copy['model_snapshots'] = []
            yaml.dump(con_copy, config_file)

        self.model_snapshots = []

    def get_dir_path(self):
        return self.dir

    def add_model_snapshot(self, name):
        path = os.path.join(self.dir, 'config.yml')

        with open(path, 'r') as file:
            data = yaml.safe_load(file)

        # Append the new snapshot name to the list
        data['model_snapshots'].append(name)

        # Write the updated data back to the file
        with open(path, 'w') as file:
            yaml.safe_dump(data, file)

    @classmethod
    def from_existing_run(cls, directory):
        instance = cls.__new__(cls)
        instance.dir = directory
        instance.name = os.path.basename(directory)
        # Assuming model snapshots are stored in a text file named 'model_snapshots.txt'
        with open(os.path.join(directory, 'config.yml'), 'r') as file:
            info=yaml.safe_load(file)
            instance.model_snapshots = info["model_snapshots"]
        return instance

    def get_snapshot(self, tensors, index=-1):
        # Check if index is out of bounds
        if index >= len(self.model_snapshots) or index < -len(self.model_snapshots):
            index = -1
        snapshot = os.path.join(self.dir, self.model_snapshots[index])
        generator = torch.load(snapshot)["model"]
        os.makedirs(os.path.join(config()["project_dir"],"test_dir"),exist_ok=True)
        process_and_save_images_as_pdf(tensors, generator, os.path.join(config()["project_dir"],"test_dir"))
