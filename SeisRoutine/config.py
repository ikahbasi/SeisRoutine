import yaml

class Config:
    def __init__(self, **entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                value = Config(**value)
            setattr(self, key, value)

    def __str__(self):
        text = ''
        for key, val in self.__dict__.items():
            text += f'{key}: {val}\n'
        return text

def load_config(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
        return Config(**config_dict)
