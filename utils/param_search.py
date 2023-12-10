import os
import operator
import csv
import numpy as np
import pandas as pd
import wandb
import itertools
from collections import defaultdict
from functools import partial, reduce


"""
modified from https://github.com/scikit-learn/scikit-learn/blob/
16625450b/sklearn/model_selection/_search.py#L1021
"""


class ParameterGrid:
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def __iter__(self):
        items = sorted(self.param_grid.items())
        if not items:
            yield {}
        else:
            keys, values = zip(*items)
            for v in itertools.product(*values):
                params = dict(zip(keys, v))
                yield params

    def __len__(self):
        product = partial(reduce, operator.mul)
        return product(len(v) for v in self.param_grid.values()) if self.param_grid else 1


class CsvWriter:
    def __init__(self, fieldnames, file_name, accuracies_dict_keys=[], acc_metrics: list = []):
        self.fieldnames = fieldnames
        self.file_name = file_name

        slash_pos = file_name.rfind('/')
        work_dir = file_name[:slash_pos]
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir, exist_ok=True)

        if not os.path.isfile(self.file_name):
            with open(self.file_name, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

        self.best_params_dict = None
        self.best_params_accs_dict = {key: 0. for key in accuracies_dict_keys}
        self.acc_metrics = acc_metrics

        if not self.acc_metrics:
            print('Using max value of accs dictionary to find best parameters')
        else:
            assert set(self.acc_metrics).issubset(accuracies_dict_keys)

    def write_row(self, row: dict):
        with open(self.file_name, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)

            assert sorted(row.keys()) == sorted(self.fieldnames)
            writer.writerow(row)

    @staticmethod
    def get_wandb_stats_table(entity: str, project: str, run_id: str):
        api = wandb.Api()
        run = api.run(entity + '/' + project + '/' + run_id)
        artifact = api.artifact(entity + '/' + project + '/run-' + run_id + '-Statstable:latest')
        wandb_table = run.use_artifact(artifact).get('Stats table')
        return wandb_table

    @staticmethod
    def wandb_table_to_stats_dict(wandb_table):
        def pretty_round(num, k=4):
            dot = num.find('.')

            if dot < k - 1:
                return num[:k + 1]
            else:
                return num[:dot]

        df = pd.DataFrame(wandb_table.data, columns=wandb_table.columns).transpose().iloc[1:]
        df = df.applymap(lambda x: pretty_round(str(x)))

        stats_col = df.iloc[:, 0].copy(deep=True)
        for i in range(1, df.shape[1]):
            stats_col += '|' + df.iloc[:, i]

        return stats_col.to_dict()

    @staticmethod
    def values_to_mean_pm_std(values, decimals=4):
        return str(np.round(np.mean(values), decimals)) + u"\u00B1" + str(np.round(np.std(values), decimals))

    def update_best_params(self, cur_accuracies_dict: dict, cur_params: dict):

        accs_dict = self.best_params_accs_dict
        assert accs_dict.keys() == cur_accuracies_dict.keys()

        if not self.acc_metrics:
            best_acc = accs_dict[max(accs_dict, key=accs_dict.get)]
            cur_best_acc = cur_accuracies_dict[max(cur_accuracies_dict, key=cur_accuracies_dict.get)]
        else:
            best_acc = max([accs_dict[key] for key in self.acc_metrics])
            cur_best_acc = max([cur_accuracies_dict[key] for key in self.acc_metrics])

        if best_acc < cur_best_acc:
            self.best_params_accs_dict = cur_accuracies_dict
            self.best_params_dict = cur_params

    def write_best_params_to_file(self):
        row = dict(**self.best_params_accs_dict, **self.best_params_dict)
        for key in self.fieldnames:
            if key not in row.keys():
                row[key] = 'None'

        self.write_row(row)
