import argparse
import importlib


def parse_args():
    parser = argparse.ArgumentParser(description='Specify config file path')
    parser.add_argument('--config', required=True, dest='config')
    args = parser.parse_args()
    return args


def get_config(args):
    spec = importlib.util.spec_from_file_location('config', args.config)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Params

