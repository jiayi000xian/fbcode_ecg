"""
This file should contain all settings specific to your dev environment and should not be committed.
Use https://fburl.com/3sxg2ez9 to make git ignore changes to this file.
"""

import os


def enable_proxy():
    # This is to make sure HF can download stuff
    os.environ["HTTP_PROXY"] = "http://fwdproxy:8080"
    os.environ["HTTPS_PROXY"] = "https://fwdproxy:8080"
    os.environ["http_proxy"] = "fwdproxy:8080"
    os.environ["https_proxy"] = "fwdproxy:8080"


# We'll specify all data locations relative to this directory
# LOCAL_DATA_DIR = "/home/danielsimig/external/ECG/data"
LOCAL_DATA_DIR = "/private/home/ledell/ECG/DATA"

# IS_MULTI_GPU = True
IS_MULTI_GPU = False
GPUS_TO_USE = [1]
# GPUS_TO_USE = [6]
DEFAULT_DEVICE = f"cuda:{GPUS_TO_USE[0]}"
