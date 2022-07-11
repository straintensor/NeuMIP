import os
import importlib.util

import importlib
import inspect


classes_dict =  {}

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    for root, dirs, files in os.walk(script_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                module_path = (os.path.join(script_dir, file))

                spec = importlib.util.spec_from_file_location("experiments." + os.path.splitext(file)[0], module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj):
                        classes_dict[name] = obj





main()

def get_experiments_list():
    return sorted(list(classes_dict.keys()))

def get_experiment(name):
    return classes_dict[name]
