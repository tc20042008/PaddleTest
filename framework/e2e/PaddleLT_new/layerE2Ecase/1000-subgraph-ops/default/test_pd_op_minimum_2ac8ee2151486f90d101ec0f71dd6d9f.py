import os
if os.getenv('FLAGS_cinn_new_group_scheduler') is None:
    os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
if os.getenv('FLAGS_group_schedule_tiling_first') is None:
    os.environ['FLAGS_group_schedule_tiling_first'] = '1'
if os.getenv('FLAGS_prim_all') is None:
    os.environ['FLAGS_prim_all'] = 'true'
if os.getenv('FLAGS_prim_enable_dynamic') is None:
    os.environ['FLAGS_prim_enable_dynamic'] = '1'
if os.getenv('FLAGS_enable_pir_api') is None:
    os.environ['FLAGS_enable_pir_api'] = '1'
if os.getenv('FLAGS_cinn_bucket_compile') is None:
    os.environ['FLAGS_cinn_bucket_compile'] = '1'

import unittest
import numpy as np
import paddle

def GetEnvVarEnableJit():
    enable_jit = os.getenv('PADDLE_DEBUG_ENABLE_JIT')
    return enable_jit not in {
        "0",
        "False",
        "false",
        "OFF",
    }

def GetEnvVarEnableCinn():
    enable_cinn = os.getenv('PADDLE_DEBUG_ENABLE_CINN')
    if enable_cinn is None:
        return True
    return enable_cinn not in {
        "0",
        "False",
        "false",
        "OFF",
    }


def GetTolerance(dtype):
    if dtype == np.float16:
        return GetFloat16Tolerance()
    if dtype == np.float32:
        return GetFloat32Tolerance()
    return 1e-6

def GetFloat16Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT16_TOL'))
    except:
        return 1e-3

def GetFloat32Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT32_TOL'))
    except:
        return 1e-6

def IsInteger(dtype):
    return np.dtype(dtype).char in np.typecodes['AllInteger']

def ApplyToStatic(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=net.get_input_spec(),
        build_strategy=build_strategy,
        full_graph=True,
    )

class InstanceTrait:

    @classmethod
    def instance(cls):
        if cls.instance_ is None:
            cls.instance_ = cls()
        return cls.instance_

    @classmethod
    def static_instance_with_cinn(cls):
        if cls.static_instance_with_cinn_ is None:
            cls.static_instance_with_cinn_ = ApplyToStatic(
                cls.instance(),
                use_cinn=True
            )
        return cls.static_instance_with_cinn_

    @classmethod
    def static_instance_without_cinn(cls):
        if cls.static_instance_without_cinn_ is None:
            cls.static_instance_without_cinn_ = ApplyToStatic(
                cls.instance(),
                use_cinn=False
            )
        return cls.static_instance_without_cinn_


class CinnTestBase:

    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def test_train(self):
        dy_outs = self.train(use_cinn=False)
        cinn_outs = self.train(use_cinn=GetEnvVarEnableCinn())

        for cinn_out, dy_out in zip(cinn_outs, dy_outs):
          if type(cinn_out) is list and type(dy_out) is list:
            for x, y in zip(cinn_out, dy_out):
              self.assert_all_close(x, y)
          else:
            self.assert_all_close(cinn_out, dy_out)

    def train(self, use_cinn):
        if GetEnvVarEnableJit():
            net = self.prepare_static_net(use_cinn)
        else:
            net = self.prepare_net()
        out = net(*self.inputs)
        return out
    
    def prepare_data(self):
        self.inputs = self.get_inputs()
        for input in self.inputs:
            input.stop_gradient = True

    def prepare_net(self):
        return self.get_test_class().instance()

    def prepare_static_net(self, use_cinn):
        if use_cinn:
            return self.get_test_class().static_instance_with_cinn()
        else:
            return self.get_test_class().static_instance_without_cinn()

    def assert_all_close(self, x, y):
        if (hasattr(x, "numpy") and hasattr(y, "numpy")):
            x_numpy = x.numpy()
            y_numpy = y.numpy()
            assert x_numpy.dtype == y_numpy.dtype
            if IsInteger(x_numpy.dtype):
                np.testing.assert_equal(x_numpy, y_numpy)
            else:
                tol = GetTolerance(x_numpy.dtype)
                np.testing.assert_allclose(x_numpy, y_numpy, atol=tol, rtol=tol)
        else:
            assert x == y



class PrimitiveOp_0066edc5450243aed008b5625d34c300(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_10cb16380a7afdf4d331e664af1e9877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10cb16380a7afdf4d331e664af1e9877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5b7b003139e13a34964dcb32bd57754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5b7b003139e13a34964dcb32bd57754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2c7910b2cb0284536640cdb8fc09d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2c7910b2cb0284536640cdb8fc09d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ff5ac1f01dbb68c2a20bf579d1397fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ff5ac1f01dbb68c2a20bf579d1397fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b523c2641be6e3a734ff3a89f7c127f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b523c2641be6e3a734ff3a89f7c127f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc85b5fb7b44ca2ebe66b7ecf956ba10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc85b5fb7b44ca2ebe66b7ecf956ba10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10cb16380a7afdf4d331e664af1e9877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_10cb16380a7afdf4d331e664af1e9877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_905ed21d1f906d3c4fd030d26dc7cddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_905ed21d1f906d3c4fd030d26dc7cddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fdd71a8ebcf89d8c60536f4c9c25694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fdd71a8ebcf89d8c60536f4c9c25694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_17a0cb282a6a20d3c31f590fb7dec118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17a0cb282a6a20d3c31f590fb7dec118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17a0cb282a6a20d3c31f590fb7dec118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17a0cb282a6a20d3c31f590fb7dec118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1841, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8149f267687e9b1bf6ad1803bc73028b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8149f267687e9b1bf6ad1803bc73028b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a04d52d9490f7cdde29a27a8978c858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a04d52d9490f7cdde29a27a8978c858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c30b0a4686cc47360bcc22ffa08c2af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c30b0a4686cc47360bcc22ffa08c2af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b451c0a3602761548de06690b6b7fb73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b451c0a3602761548de06690b6b7fb73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b443771718ef0e9afd386445d763529e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.48502784967422485], [0.007254623807966709], [0.08551525324583054], [0.28749170899391174], [0.42672285437583923], [0.4068581461906433], [0.00696443160995841], [0.11411301791667938], [0.016163386404514313]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.22089730203151703], [0.033434219658374786], [0.16276678442955017], [0.3074219524860382], [0.03921782970428467], [0.38058915734291077], [0.19385917484760284], [0.09035753458738327], [0.05179273337125778]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_5b053359acd8e3a1b507745c49ba8877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43891218304634094], [0.3814011812210083], [0.4228847026824951], [0.03740933537483215], [0.2268114984035492], [0.26206380128860474], [0.4565003216266632], [0.3727211654186249], [0.10162022709846497]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.032230887562036514], [0.18736937642097473], [0.015003582462668419], [0.33160749077796936], [0.09220337867736816], [0.21914060413837433], [0.005615626461803913], [0.10680714249610901], [0.24876168370246887]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_a94485c30d55a5f75a5a7d7d225d50f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2136506289243698], [0.44700920581817627], [0.4920419454574585], [0.324871301651001], [0.4347423315048218], [0.11308283358812332], [0.4580673575401306], [0.2829873263835907], [0.09454172104597092]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.05196747928857803], [0.3380136489868164], [0.4891405999660492], [0.2964899241924286], [0.20674164593219757], [0.010906912386417389], [0.49398526549339294], [0.1478980928659439], [0.371082603931427]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_b5599d48497a8e4b8a816d827b0ad998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3358563482761383], [0.32271942496299744], [0.44343236088752747], [7.760437438264489e-05], [0.2671041786670685], [0.39811384677886963], [0.4129534661769867], [0.08367976546287537], [0.2420433908700943]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.4841810464859009], [0.08505020290613174], [0.18903541564941406], [0.38031214475631714], [0.10913760215044022], [0.46841099858283997], [0.29409098625183105], [0.2900417149066925], [0.05474267899990082]], dtype='float32').reshape([9, 1]),
        ]


class TestPrimitiveOp_290fcca72fd568c0b9d3f665d1175192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290fcca72fd568c0b9d3f665d1175192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290fcca72fd568c0b9d3f665d1175192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_290fcca72fd568c0b9d3f665d1175192(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5562, 1], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_108a2592e54ab6ece55bbdc4731baeb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14741972088813782, 0.37168705463409424, 0.3194563388824463, 0.4567588269710541, 0.24708279967308044, 0.33666518330574036], dtype='float32').reshape([6]),
            paddle.to_tensor([0.22594380378723145, 0.10565175116062164, 0.03872787579894066, 0.2721693217754364, 0.14351142942905426, 0.3796778917312622], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_d0180ff3f513293a41e26fd3c3423735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3068593740463257, 0.12108806520700455, 0.2023414820432663, 0.36508259177207947, 0.45763009786605835, 0.3360748291015625], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1421709507703781, 0.1708022505044937, 0.1709064394235611, 0.046536415815353394, 0.06752748787403107, 0.041682589799165726], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_7792bfceea62dc50a4d0b5044d6cd1b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14741972088813782, 0.1336534470319748, 0.11528350412845612, 0.4567588269710541, 0.24708279967308044, 0.3292865753173828], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2984810769557953, 0.018444553017616272, 0.06930149346590042, 0.3149198293685913, 0.005490172654390335, 0.3335161805152893], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_1fc6887971d455089be224f73b7b06f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9d8ec59a9524c49ea98c653ec112e69
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25797852873802185, 0.12108806520700455, 0.2023414820432663, 0.36508259177207947, 0.0434037484228611, 0.07065112143754959], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2929052710533142, 0.40369516611099243, 0.1233789473772049, 0.3182181119918823, 0.267822802066803, 0.4557313323020935], dtype='float32').reshape([6]),
        ]


class TestPrimitiveOp_4cd71d737ca1f111ae6eee194c26bde8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cd71d737ca1f111ae6eee194c26bde8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cd71d737ca1f111ae6eee194c26bde8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4cd71d737ca1f111ae6eee194c26bde8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1734, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ff5ac1f01dbb68c2a20bf579d1397fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ff5ac1f01dbb68c2a20bf579d1397fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a04d52d9490f7cdde29a27a8978c858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a04d52d9490f7cdde29a27a8978c858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc3f7cea898295af919a63cda9217ac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc3f7cea898295af919a63cda9217ac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc3f7cea898295af919a63cda9217ac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc3f7cea898295af919a63cda9217ac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1541, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc85b5fb7b44ca2ebe66b7ecf956ba10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dc85b5fb7b44ca2ebe66b7ecf956ba10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5b7b003139e13a34964dcb32bd57754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e5b7b003139e13a34964dcb32bd57754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e2066a93e8739cbd7a5f0c6e34bb983(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30370238423347473]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.17654746770858765]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_1379ed0d82bb8acc7b0438888902b43b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.34866175055503845]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.11643274873495102]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_b0235e08f18c0eb81c54409d9204e0e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26397430896759033]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.46222788095474243]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_80c602eb1d00fe041c852d61d49455c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.25180718302726746]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.13167576491832733]], dtype='float32').reshape([1, 1]),
        ]


class TestPrimitiveOp_ee4abe1bde0213bace08d6414a5704ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.332610547542572], [0.14597834646701813], [0.19613602757453918], [0.12908796966075897], [0.26270008087158203], [0.133039191365242]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3089545667171478], [0.24098819494247437], [0.3093312382698059], [0.2416381686925888], [0.2358388751745224], [0.31183764338493347]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_fb5f335ce2dfb4190388cee2a2b16665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40432968735694885], [0.2534337639808655], [0.18960238993167877], [0.41409391164779663], [0.25769028067588806], [0.23563654720783234]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.07759702950716019], [0.23825453221797943], [0.15273594856262207], [0.2640349566936493], [0.366540789604187], [0.189493328332901]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_a0bc6b396b0761d79b166c595290ea44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3404685854911804], [0.3644312620162964], [0.44551825523376465], [0.496250182390213], [0.2719564139842987], [0.07249618321657181]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.03989572077989578], [0.06823953241109848], [0.13498272001743317], [0.24980278313159943], [0.27458834648132324], [0.18564586341381073]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_99a392bbfdf8b48baacb677dbcd6c864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3674864172935486], [0.35537904500961304], [0.435450941324234], [0.12282329052686691], [0.16958539187908173], [0.04632904753088951]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.44728875160217285], [0.1407179981470108], [0.16133081912994385], [0.4250192940235138], [0.3090977668762207], [0.028634952381253242]], dtype='float32').reshape([6, 1]),
        ]


class TestPrimitiveOp_8149f267687e9b1bf6ad1803bc73028b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8149f267687e9b1bf6ad1803bc73028b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2c7910b2cb0284536640cdb8fc09d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2c7910b2cb0284536640cdb8fc09d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ff85653c03a2888a62076adfe4a4892(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ff85653c03a2888a62076adfe4a4892(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ff85653c03a2888a62076adfe4a4892(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ff85653c03a2888a62076adfe4a4892(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2061, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8e8f5ae9bc50034fdd6f600553a7fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8e8f5ae9bc50034fdd6f600553a7fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f13789372a82f54e638479a18ed06ef2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f13789372a82f54e638479a18ed06ef2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f13789372a82f54e638479a18ed06ef2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f13789372a82f54e638479a18ed06ef2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4642, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c16d4a8b8176247f5cc624ef20d2554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c16d4a8b8176247f5cc624ef20d2554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c16d4a8b8176247f5cc624ef20d2554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c16d4a8b8176247f5cc624ef20d2554(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b523c2641be6e3a734ff3a89f7c127f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b523c2641be6e3a734ff3a89f7c127f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c30b0a4686cc47360bcc22ffa08c2af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c30b0a4686cc47360bcc22ffa08c2af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d7f2691e65eea571313e64dab29a5db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4107484221458435], [0.2350795716047287], [0.32436296343803406], [0.33859461545944214], [0.17683643102645874]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.00047272868687286973], [0.056612178683280945], [0.17818866670131683], [0.2440360188484192], [0.006049283314496279]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_9406fb29304c7e278036d0aadf0b9d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2710123062133789], [0.24374403059482574], [0.10803781449794769], [0.17378324270248413], [0.29924002289772034]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.049238573759794235], [0.14470472931861877], [0.00861063040792942], [0.40792182087898254], [0.3668667674064636]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_197850db4d5b4e6141dac820c1856464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1571986973285675], [0.11458301544189453], [0.34768521785736084], [0.20442333817481995], [0.04476496949791908]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.008164001628756523], [0.1091982051730156], [0.34744080901145935], [0.31836310029029846], [0.16771680116653442]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_90ae01b10e4c36f3493f6113b97c0aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3675876557826996], [0.2628155052661896], [0.3304205536842346], [0.4551815986633301], [0.469809353351593]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2649228572845459], [0.3236313760280609], [0.3462056517601013], [0.14429233968257904], [0.2508985102176666]], dtype='float32').reshape([5, 1]),
        ]


class TestPrimitiveOp_8fdd71a8ebcf89d8c60536f4c9c25694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fdd71a8ebcf89d8c60536f4c9c25694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b451c0a3602761548de06690b6b7fb73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b451c0a3602761548de06690b6b7fb73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcec9f6bda8fecd4654627c6235699d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcec9f6bda8fecd4654627c6235699d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb589b1533164b421e7f4207bb08825e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb589b1533164b421e7f4207bb08825e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb589b1533164b421e7f4207bb08825e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb589b1533164b421e7f4207bb08825e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2369, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e8e0c28482a5bbdd37332678cd498b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e8e0c28482a5bbdd37332678cd498b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e8e0c28482a5bbdd37332678cd498b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e8e0c28482a5bbdd37332678cd498b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3054, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7544df94e043674f2d4e92fb0e341b5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7544df94e043674f2d4e92fb0e341b5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7544df94e043674f2d4e92fb0e341b5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7544df94e043674f2d4e92fb0e341b5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3819, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8e8f5ae9bc50034fdd6f600553a7fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8e8f5ae9bc50034fdd6f600553a7fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f66c6b4f8815a4fbec4153225e7dcc8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3838992714881897], [0.1253899484872818], [0.24463814496994019], [0.23376497626304626]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.27045926451683044], [0.49347126483917236], [0.4834122955799103], [0.4336952865123749]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_79bea752f0e6bbf6b928840dd2a4592f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19007621705532074], [0.4756835699081421], [0.2541756331920624], [0.4143301248550415]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.04296104982495308], [0.3528343439102173], [0.31736236810684204], [0.22007668018341064]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_a007ad38884f0d7fc699e0d50b92fd7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06872285902500153], [0.06164400279521942], [0.20687055587768555], [0.3251105844974518]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.4831085503101349], [0.4750896692276001], [0.1859712302684784], [0.05235862359404564]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_04c2f4056447637cee36372a1a6a43f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e098a4a1a4457c33ab52971f06001bb6
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.32444852590560913], [0.14037735760211945], [0.08293063938617706], [0.08842404931783676]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.31226545572280884], [0.035297587513923645], [0.2893640995025635], [0.003905576653778553]], dtype='float32').reshape([4, 1]),
        ]


class TestPrimitiveOp_b9fd9eb2b79e1d0a097e3937bfff40bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b9fd9eb2b79e1d0a097e3937bfff40bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b9fd9eb2b79e1d0a097e3937bfff40bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b9fd9eb2b79e1d0a097e3937bfff40bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2092, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcec9f6bda8fecd4654627c6235699d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcec9f6bda8fecd4654627c6235699d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_905ed21d1f906d3c4fd030d26dc7cddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_905ed21d1f906d3c4fd030d26dc7cddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e88bc0cd04d415e65a266a1a8b5b37c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e88bc0cd04d415e65a266a1a8b5b37c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b66b44ada9a0fe1dd7f3fa7a9625a4e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b66b44ada9a0fe1dd7f3fa7a9625a4e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b66b44ada9a0fe1dd7f3fa7a9625a4e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b66b44ada9a0fe1dd7f3fa7a9625a4e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f99ba83f8f250754b59e109499c9f3a8
    def get_inputs(self):
        return [
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4214, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e88bc0cd04d415e65a266a1a8b5b37c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e88bc0cd04d415e65a266a1a8b5b37c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0066edc5450243aed008b5625d34c300
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()