import GP_force_predictor as gp
import os
import re


def train_gp(fpath, data_name_str):
    for path, subdirs, files in os.walk(fpath):
        for name in files:
            if data_name_str.lower() in name.lower():
                cc = re.search('config_(\d+)', name, re.IGNORECASE)
                if cc:
                    print('learning data: ')
                    print(name)
                    try:
                        gp.gp_model(path, name, cc.group(1))
                    except IndexError:
                        print('Error in fitting ' + name + '. Carrying on...')
                    except AttributeError:
                        print('Error in finding ' + name + '. Carrying on...')
