import os
import tempfile

from src import utils


# pylint: disable=all
def test_store_load() -> None:
    _, path = tempfile.mkstemp()
    data = {
        "string": "Hello there",
        "int": 42,
        "float": 3.141,
        "tuple": (1, 2, 3, "4"),
        "list": [1, 2, 3],
    }
    try:
        utils.store(data, path)

        loaded = utils.load(path)

        assert loaded == data  # nosec
    finally:
        os.remove(path)
