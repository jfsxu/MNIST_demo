#
# Utilities function to load YAML config files for general settings
#
import os
import yaml

def load_error_codes(config_folder_path):
    error_codes_file = os.path.join(config_folder_path, 'errors.yml')
    with open(error_codes_file) as stream:
        try:
            error_codes = yaml.safe_load(stream)
            return error_codes
        except yaml.YAMLError as exc:
            print('Can\'t load error.yml')
            exit(1)

def load_config(config_folder_path):
    errorCodes = load_error_codes(config_folder_path)
    config_file = os.path.join(config_folder_path, 'config.yml')
    with open(config_file) as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)
            print (errorCodes["CONFIG_FILE_NOT_FOUND"]["desc"])
            exit(errorCodes["CONFIG_FILE_NOT_FOUND"]["code"])