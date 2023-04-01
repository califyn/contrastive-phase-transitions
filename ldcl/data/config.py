import json

from munch import DefaultMunch

def read_config(f):
    """
        Read config files. Implement this in a function in case we need to change this at some point.

        We should probably move this somewhere else at some point, as the training loop will need it as well.

        :param: f: path to config file to be read
        :return: x: an object with attributes that are the defined parameters
    """

    with open(f, "r") as stream:
        x = json.load(stream)

    def convert_keys(d):
        for key, value in d.items():
            if value in ["None", "True", "False"]:
                d[key] = eval(d[key])
            elif isinstance(value, str) and "," in value:
                d[key] = eval("[" + d[key] + "]")
            elif isinstance(value, dict):
                convert_keys(value)
            elif isinstance(value, list) and all([isinstance(x, dict) for x in value]):
                for i, x in enumerate(value):
                    convert_keys(x)

    convert_keys(x)

    return DefaultMunch.fromDict(x, object())

