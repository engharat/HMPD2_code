# import pyyaml module
import yaml
from yaml.loader import SafeLoader

class yamlManager():

    def __init__(self, file, banckmarkhistoryFile):
        self.file = file
        self.banckmarkhistoryFile = banckmarkhistoryFile
        self.data = None

    def read_conf(self):

        # Open the file and load the file
        with open(self.file) as f:
            self.data = yaml.load(f, Loader=SafeLoader)
            self.save_conf()
            self.formatt_data()
            return self.data

    def checkKey(self, dic, key):
        if key in dic.keys():
            return True
        else:
            return False

    # transform the list in the tuple type desired by the Pytorch transform module
    def formatt_data(self):
        tuple_params = ['transform_crop', 'transform_normalize_mean', 'transform_normalize_var', 'transform_resize']
        for param in tuple_params:
            if self.checkKey(self.data, param):
                self.data[param] = tuple(self.data[param])


    def save_conf(self):
        with open(self.banckmarkhistoryFile, 'w') as f:
            data = yaml.dump(self.data, f, sort_keys=False, default_flow_style=False)
