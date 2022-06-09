""" Generates the tfdv schema from data """


import os

import pandas as pd
import tensorflow_data_validation as tfdv

from lib import conf


def main() -> None:
    """
    Generates the schema from a tsv file and stores the schema.
    """
    dataframe = pd.read_csv(os.path.join(conf.PROCESSED_DATA_DIR, "preprocessed.csv"))
    stats = tfdv.generate_statistics_from_dataframe(dataframe)
    schema = tfdv.infer_schema(statistics=stats)
    tfdv.write_schema_text(schema, os.path.join(conf.DATA_STATS_DIR, "schema.txt"))


if __name__ == "__main__":
    main()
