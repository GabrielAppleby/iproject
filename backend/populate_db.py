import pathlib
import psycopg2
from typing import List, Tuple

import numpy as np

from app.core.data_instance import DataInstance

CURRENT_PATH: pathlib.Path = pathlib.Path(__file__).parent.absolute()
RAW_DATA_PATH: pathlib.Path = CURRENT_PATH.joinpath('raw_data')

IRIS = 'iris'
WINE = 'wine'
ECOLI = 'ecoli'

FLOWERS = 'flowers'
BACTERIA = 'bacteria'

DB_FILE_TEMPLATE = 'db_payload_{}.npz'


def load_data(name):
    data = np.load(pathlib.Path(RAW_DATA_PATH, DB_FILE_TEMPLATE.format(name)))
    features = data['features']
    targets = data['targets']

    return features, targets


def read_data(features, targets) -> List[DataInstance]:
    instances = []
    for feature, target in zip(features, targets):
        instances.append(DataInstance(0,
                                      feature,
                                      float(target[0])))

    return instances


def main():
    datasets: List[Tuple] = [(IRIS, FLOWERS)]
                             # (WINE, WINE),
                             # (ECOLI, BACTERIA)]

    for dataset_name, db_table_name in datasets:
        features, targets = load_data(dataset_name)
        instances = read_data(features, targets)
        connection = psycopg2.connect(user="dev_user",
                                      password="dev_pass",
                                      host="db",
                                      port="5432",
                                      database="dev_db")
        c = connection.cursor()
        for instance in instances:
            c.execute(
                'INSERT INTO flowers (features, target)'
                ' VALUES (%s, %s)',
                (instance.features.tobytes(), instance.target)
            )
        connection.commit()
        connection.close()


if __name__ == '__main__':
    main()
