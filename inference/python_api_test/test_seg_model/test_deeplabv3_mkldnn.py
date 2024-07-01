# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test deeplabv3 model
"""

import os
import sys
import logging
import tarfile
import six
import wget
import pytest
import numpy as np

# pylint: disable=wrong-import-position
sys.path.append("..")
from test_case import InferenceTest


# pylint: enable=wrong-import-position


def check_model_exist():
    """
    check model exist
    """
    deeplabv3_url = "https://paddle-qa.bj.bcebos.com/inference_model/2.6/seg/deeplabv3.tgz"
    if not os.path.exists("./deeplabv3/model.pdiparams"):
        wget.download(deeplabv3_url, out="./")
        tar = tarfile.open("deeplabv3.tgz")
        tar.extractall()
        tar.close()


def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./deeplabv3/model.pdmodel",
        params_file="./deeplabv3/model.pdiparams",
    )
    test_suite.config_test()


@pytest.mark.win
@pytest.mark.server
@pytest.mark.mkldnn_bz1_precision
def test_mkldnn():
    """
    compared mkldnn batch_size=1 deeplabv3 outputs with true val
    """
    check_model_exist()

    file_path = "./deeplabv3"
    images_size = 224
    batch_size = 1
    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./deeplabv3/model.pdmodel",
        params_file="./deeplabv3/model.pdiparams",
    )
    images_list, npy_list = test_suite.get_images_npy(file_path, images_size)
    fake_input = np.array(images_list[0:batch_size]).astype("float32")
    input_data_dict = {"x": fake_input}
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="cpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(
        model_file="./deeplabv3/model.pdmodel",
        params_file="./deeplabv3/model.pdiparams",
    )
    test_suite2.mkldnn_test(input_data_dict, output_data_dict)

    del test_suite2  # destroy class to save memory