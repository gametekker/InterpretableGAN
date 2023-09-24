import yaml
def config():
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    return config
def hyperparameters():
    with open('hyperparameters.yml', 'r') as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters
