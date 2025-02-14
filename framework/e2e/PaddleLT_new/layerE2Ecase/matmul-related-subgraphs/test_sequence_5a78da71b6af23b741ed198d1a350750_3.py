import os
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'
import sys
import unittest
import numpy as np
from dataclasses import dataclass
import typing as t
import itertools

@dataclass
class Stage:
    name: str
    env_vars: t.Dict[str, str]

cinn_stages = [
    Stage(
        name="dynamic_to_static",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=False,
            FLAGS_prim_all=False,
            FLAGS_prim_enable_dynamic=False,
        ),
    ),
    Stage(
        name="prim",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=False,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
        ),
    ),
    Stage(
        name="infer_symbolic",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=False,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=False,
            FLAGS_check_infer_symbolic=True,
        ),
    ),
	Stage(
        name="frontend",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=True,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=True,
            FLAGS_check_infer_symbolic=False,
            FLAGS_enable_fusion_fallback=True,
        ), 
    ),
    Stage(
        name="backend",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=True,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=True,
            FLAGS_check_infer_symbolic=False,
            FLAGS_enable_fusion_fallback=False,
        ), 
    ),
]

def GetCinnStageByName(name):
    for stage in cinn_stages:
        if stage.name == name:
            return stage
    return None

def GetCurrentCinnStage():
    name = os.getenv('PADDLE_DEBUG_CINN_STAGE_NAME')
    if name is None:
        return None
    stage_names = [stage.name for stage in cinn_stages]
    assert name in stage_names, (
        f"PADDLE_DEBUG_CINN_STAGE_NAME should be in {stage_names}"
    )
    return GetCinnStageByName(name)

def GetPrevCinnStage(stage):
    for i in range(1, len(cinn_stages)):
        if stage is cinn_stages[i]:
            return cinn_stages[i - 1]
    return None

def IsCinnStageEnableDiff():
    value = os.getenv('PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF')
    enabled = value in {
        '1',
        'true',
        'True',
    }
    if enabled:
        assert GetCurrentCinnStage() is not None
    return enabled

def GetExitCodeAndStdErr(cmd, env):
    env = {
        k:v
        for k, v in env.items()
        if v is not None
    }
    import subprocess
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return result.returncode, result.stderr

def GetStageExitCodeAndStdErr(stage):
    return GetExitCodeAndStdErr(
        [sys.executable, __file__],
        env=dict(
            PADDLE_DEBUG_CINN_STAGE_NAME=stage.name,
            PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF='0',
            PYTHONPATH=os.getenv('PYTHONPATH'),
            ATHENA_ENABLE_TRY_RUN="False",
        ),
    )

def AthenaTryRunEnabled():
    return os.getenv('ATHENA_ENABLE_TRY_RUN') not in {
        "0",
        "False",
        "false",
        "OFF"
    }

def GetNeedSkipAndSkipMessage():
    current_stage = GetCurrentCinnStage()
    assert current_stage is not None
    if not IsCinnStageEnableDiff():
        return False, ""
    last_stage = GetPrevCinnStage(current_stage)
    if last_stage is None:
        return False, ""
    exitcode, stderr = GetStageExitCodeAndStdErr(last_stage)
    if exitcode != 0:
        return True, "last stage failed."
    return False, ""

def GetCurrentStageTryRunExitCodeAndStdErr():
    if not AthenaTryRunEnabled():
        return False, ""
    current_stage = GetCurrentCinnStage()
    assert current_stage is not None
    return GetStageExitCodeAndStdErr(current_stage)

def SetDefaultEnv(**env_var2value):
    for env_var, value in env_var2value.items():
        if os.getenv(env_var) is None:
            os.environ[env_var] = str(value)

SetDefaultEnv(
    PADDLE_DEBUG_CINN_STAGE_NAME="backend",
    PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF=False,
    PADDLE_DEBUG_ENABLE_CINN=True,
    FLAGS_enable_pir_api=True,
    FLAGS_prim_all=True,
    FLAGS_prim_enable_dynamic=True,
    FLAGS_use_cinn=False,
    FLAGS_check_infer_symbolic=False,
    FLAGS_enable_fusion_fallback=False,
)

import paddle

def SetEnvVar(env_var2value):
    for env_var, value in env_var2value.items():
        os.environ[env_var] = str(value)
    paddle.set_flags({
        env_var:value
        for env_var, value in env_var2value.items()
        if env_var.startswith('FLAGS_')
    })

if GetCurrentCinnStage() is not None:
    SetEnvVar(GetCurrentCinnStage().env_vars)

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

    def _test_entry(self):
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
        paddle.seed(2024)
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





need_skip, skip_message = GetNeedSkipAndSkipMessage()
try_run_exit_code, try_run_stderr = GetCurrentStageTryRunExitCodeAndStdErr()
class TestTryRun(unittest.TestCase):
    def test_panic(self):
        if not AthenaTryRunEnabled():
            return
        if try_run_exit_code == 0:
            # All unittest cases passed.
            return
        if try_run_exit_code > 0:
            # program failed but not panic.
            return
        # program panicked.
        kOutputLimit = 65536
        message = try_run_stderr[-kOutputLimit:]
        raise RuntimeError(f"panicked. last {kOutputLimit} characters of stderr: \n{message}")

class SubGraphLayer(InstanceTrait, paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, t0, t1, t2, t3, t4):
        # pd_op.shape64: (3xi64) <- (-1x49x384xf32)
        t5 = paddle._C_ops.shape64(t0)
        
        # pd_op.full_int_array: (1xi64) <- ()
        t6 = [0]
        
        # pd_op.assign: (1xi64) <- (1xi64)
        t7 = t6
        
        # pd_op.assign: (1xi64) <- (1xi64)
        t8 = t6
        
        # pd_op.full_int_array: (1xi64) <- ()
        t9 = [1]
        
        # pd_op.assign: (1xi64) <- (1xi64)
        t10 = t9
        
        # pd_op.assign: (1xi64) <- (1xi64)
        t11 = t9
        
        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t12 = paddle._C_ops.slice(t5, [0], t6, t9, [1], [0])
        del t5
        
        # pd_op.matmul: (-1x49x1152xf32) <- (-1x49x384xf32, 384x1152xf32)
        t13 = paddle._C_ops.matmul(t0, t1, False, False)
        del t0, t1
        
        # pd_op.add: (-1x49x1152xf32) <- (-1x49x1152xf32, 1152xf32)
        t14 = paddle._C_ops.add(t13, t2)
        del t2
        
        # pd_op.full_int_array: (5xi64) <- ()
        t15 = [-1, 49, 3, 12, 32]
        
        # pd_op.reshape: (-1x49x3x12x32xf32) <- (-1x49x1152xf32, 5xi64)
        t16 = paddle._C_ops.reshape(t14, t15)
        del t15
        
        # pd_op.transpose: (3x-1x12x49x32xf32) <- (-1x49x3x12x32xf32)
        t17 = paddle._C_ops.transpose(t16, [2, 0, 3, 1, 4])
        del t16
        
        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        t18 = paddle._C_ops.slice(t17, [0], t6, t9, [1], [0])
        
        # pd_op.full_int_array: (1xi64) <- ()
        t19 = [2]
        
        # pd_op.assign: (1xi64) <- (1xi64)
        t20 = t19
        
        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        t21 = paddle._C_ops.slice(t17, [0], t9, t19, [1], [0])
        
        # pd_op.full_int_array: (1xi64) <- ()
        t22 = [3]
        
        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        t23 = paddle._C_ops.slice(t17, [0], t19, t22, [1], [0])
        
        # pd_op.full: (1xf32) <- ()
        t24 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())
        
        # pd_op.scale: (-1x12x49x32xf32) <- (-1x12x49x32xf32, 1xf32)
        t25 = paddle._C_ops.scale(t18, t24, float('0'), True)
        del t18
        
        # pd_op.transpose: (-1x12x32x49xf32) <- (-1x12x49x32xf32)
        t26 = paddle._C_ops.transpose(t21, [0, 1, 3, 2])
        del t21
        
        # pd_op.matmul: (-1x12x49x49xf32) <- (-1x12x49x32xf32, -1x12x32x49xf32)
        t27 = paddle._C_ops.matmul(t25, t26, False, False)
        
        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        t28 = paddle._C_ops.flatten(t3, 0, 1)
        del t3
        
        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        t29 = paddle._C_ops.index_select(t4, t28, 0)
        del t4
        
        # pd_op.full_int_array: (3xi64) <- ()
        t30 = [49, 49, -1]
        
        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        t31 = paddle._C_ops.reshape(t29, t30)
        del t30
        
        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        t32 = paddle._C_ops.transpose(t31, [2, 0, 1])
        del t31
        
        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        t33 = paddle._C_ops.unsqueeze(t32, t6)
        
        # pd_op.add: (-1x12x49x49xf32) <- (-1x12x49x49xf32, 1x12x49x49xf32)
        t34 = paddle._C_ops.add(t27, t33)
        
        return t6, t7, t8, t9, t10, t11, t13, t14, t17, t19, t20, t22, t23, t24, t25, t26, t27, t28, t29, t32, t33, t34

    def get_input_spec(self):
        return [
            # t0
            paddle.static.InputSpec(shape=[None, 49, 384], dtype='float32'),
            # t1
            paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
            # t2
            paddle.static.InputSpec(shape=[1152], dtype='float32'),
            # t3
            paddle.static.InputSpec(shape=[49, 49], dtype='int64'),
            # t4
            paddle.static.InputSpec(shape=[169, 12], dtype='float32'),
        ]

    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestSubGraphLayer(CinnTestBase, unittest.TestCase):
    def get_test_class(self):
        return SubGraphLayer

    def get_inputs(self):
        return [
            # t0
            paddle.uniform([96, 49, 384], dtype='float32', min=0, max=0.5),
            # t1
            paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            # t2
            paddle.uniform([1152], dtype='float32', min=0, max=0.5),
            # t3
            paddle.cast(paddle.randint(low=0, high=3, shape=[49, 49], dtype='int64'), 'int64'),
            # t4
            paddle.uniform([169, 12], dtype='float32', min=0, max=0.5),
        ]

    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

if __name__ == '__main__':
    unittest.main()