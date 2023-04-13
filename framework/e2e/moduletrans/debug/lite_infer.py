#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
aaa
"""

from paddlelite.lite import *
import numpy as np

# 1. Set config information
config = MobileConfig()
# 2. Set the path to the model generated by opt tools
config.set_model_from_file("ConvBNLayer_opt.nb")
# 3. Create predictor by config
predictor = create_paddle_predictor(config)


input_tensor = predictor.get_input(0)
input_tensor.from_numpy(np.ones((1, 3, 224, 224)).astype("float32"))

predictor.run()

output_tensor = predictor.get_output(0)
output_data = output_tensor.numpy()
print(output_data)