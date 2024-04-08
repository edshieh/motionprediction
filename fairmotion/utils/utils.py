# Copyright (c) Facebook, Inc. and its affiliates.

import math
import numpy as np
import os
import random
import torch
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from typing import Any, Callable, Dict, Iterable, List, Tuple


def str_to_axis(s: str) -> np.ndarray:
    if s == "x":
        return np.array([1.0, 0.0, 0.0])
    elif s == "y":
        return np.array([0.0, 1.0, 0.0])
    elif s == "z":
        return np.array([0.0, 0.0, 1.0])
    else:
        raise Exception


def axis_to_str(a: np.ndarray) -> str:
    if np.array_equal(a, [1.0, 0.0, 0.0]):
        return "x"
    elif np.array_equal(a, [0.0, 1.0, 0.0]):
        return "y"
    elif np.array_equal(a, [0.0, 0.0, 1.0]):
        return "z"
    else:
        raise Exception


def get_index(index_dict: Dict, key: int|str|Any):
    if isinstance(key, int):
        return key
    elif isinstance(key, str):
        return index_dict[key]
    else:
        return index_dict[key.name]


def run_parallel(func: Callable, iterable: Iterable, num_cpus: int=20, **kwargs) -> List:
    """
    Run function over multiple cpus. The function must be written such that
    it processes single input value.

    Args:
        func: Method that is run in parallel. The first argument of func
            accepts input values from iterable.
        iterable: List of input values that func is executed over.
        num_cpus: Number of cpus used by multiprocessing.
        kwargs: Dictionary of keyword arguments that is passed on to each
            parallel call to the function

    Returns:
        Flattened list of results from running the function on iterable
        arguments
    """
    func_with_kwargs = partial(func, **kwargs)
    with Pool(processes=num_cpus) as pool:
        results = pool.map(func_with_kwargs, iterable)
    return results


def files_in_dir(
    path,
    ext=None,
    keyword=None,
    sort=False,
    sample_mode=None,
    sample_num=None,
    keywords_exclude=[],
) -> List:
    """Returns list of files in `path` directory.

    Args:
        path: Path to directory to list files from
        ext: Extension of files to be listed
        keyword: Return file if filename contains `keyword`
        sort: Sort files by filename in the returned list
        sample_mode: str; Use this option to return subset of files from `path`
            directory. `sample_mode` takes values 'sequential' to return first
            `sample_num` files, or 'shuffle' to return `sample_num` number of
            files randomly
        sample_num: Number of files to return
        exclude: the files in this this are excluded
    """
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            add = True
            if ext is not None and not file.endswith(ext):
                add = False
            if keyword is not None and keyword not in file:
                add = False
            for ke in keywords_exclude:
                if ke in file:
                    add = False
                    break
            if add:
                files.append(os.path.join(r, file))
    if sort:
        files.sort()

    if sample_num is None:
        sample_num = len(files)
    else:
        sample_num = min(sample_num, len(files))

    if sample_mode is None:
        pass
    elif sample_mode == "sequential":
        files = files[:sample_num]
    elif sample_mode == "shuffle":
        files = random.shuffle(files)[:sample_num]
    else:
        raise NotImplementedError

    return files


def _apply_fn_agnostic_to_vec_mat(input, fn):
    output = np.array([input]) if input.ndim == 1 else input
    output = np.apply_along_axis(fn, 1, output)
    return output[0] if input.ndim == 1 else output


def create_dir_if_absent(path: str|Path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_device(arg_device: str):
    # if user requested a specific type then set that
    if arg_device:
        return arg_device

    # Check if gpu backend is available otherwise use cpu
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    return device


def yes_no_input(question: str="Enter yes/y or no/n: "):
    """Simple recursive to prompt user for yes or no response

    Args:
        question (str, optional): Question to prompt user with. Defaults to "Enter yes/y or no/n: ".

    Returns:
        bool: whether the user has responded with yes (True) or no (False)
    """
    response = input(question).lower().strip()
    if response in ("yes", "y"):
        return True
    elif response in ("no", "n"):
        return False
    else:
        return yes_no_input("Please enter yes/y or no/n: ")


def convert_byte_to_humanreadable(size_bytes: str|int):
    # convert to int if str
    if isinstance(size_bytes, str):
        size_bytes = int(size_bytes)
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])