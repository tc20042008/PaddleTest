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

    def forward(self, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15):
        t6 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        t7 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        t13 = paddle._C_ops.full_int_array([-1, 7, 7, 192], paddle.int64, paddle.core.CPUPlace())
        t14 = paddle._C_ops.full_int_array([-1, 4, 4, 7, 7, 192], paddle.int64, paddle.core.CPUPlace())
        t15 = paddle._C_ops.full_int_array([-1, 28, 28, 192], paddle.int64, paddle.core.CPUPlace())
        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        t16 = [t0, t1, t2, t3, t3]
        del t0, t1
        
        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        t17 = paddle._C_ops.stack(t16, 0)
        del t16
        
        # pd_op.reshape: (-1x16x6x49x49xf32) <- (-1x6x49x49xf32, 5xi64)
        t18 = paddle._C_ops.reshape(t4, t17)
        del t4, t17
        
        # pd_op.unsqueeze: (16x1x49x49xf32) <- (16x49x49xf32, 1xi64)
        t19 = paddle._C_ops.unsqueeze(t5, t6)
        del t5
        
        # pd_op.unsqueeze: (1x16x1x49x49xf32) <- (16x1x49x49xf32, 1xi64)
        t20 = paddle._C_ops.unsqueeze(t19, t7)
        del t19
        
        # pd_op.add: (-1x16x6x49x49xf32) <- (-1x16x6x49x49xf32, 1x16x1x49x49xf32)
        t21 = paddle._C_ops.add(t18, t20)
        del t18, t20
        
        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t22 = [t8, t2, t3, t3]
        del t2
        
        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t23 = paddle._C_ops.stack(t22, 0)
        del t22
        
        # pd_op.reshape: (-1x6x49x49xf32) <- (-1x16x6x49x49xf32, 4xi64)
        t24 = paddle._C_ops.reshape(t21, t23)
        del t21, t23
        
        # pd_op.softmax: (-1x6x49x49xf32) <- (-1x6x49x49xf32)
        t25 = paddle._C_ops.softmax(t24, -1)
        del t24
        
        # pd_op.matmul: (-1x6x49x32xf32) <- (-1x6x49x49xf32, -1x6x49x32xf32)
        t26 = paddle._C_ops.matmul(t25, t9, False, False)
        del t9, t25
        
        # pd_op.transpose: (-1x49x6x32xf32) <- (-1x6x49x32xf32)
        t27 = paddle._C_ops.transpose(t26, [0, 2, 1, 3])
        del t26
        
        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t28 = [t8, t3, t10]
        del t8
        
        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t29 = paddle._C_ops.stack(t28, 0)
        del t28
        
        # pd_op.reshape: (-1x49x192xf32) <- (-1x49x6x32xf32, 3xi64)
        t30 = paddle._C_ops.reshape(t27, t29)
        del t29, t27
        
        # pd_op.matmul: (-1x49x192xf32) <- (-1x49x192xf32, 192x192xf32)
        t31 = paddle._C_ops.matmul(t30, t11, False, False)
        del t11, t30
        
        # pd_op.add: (-1x49x192xf32) <- (-1x49x192xf32, 192xf32)
        t32 = paddle._C_ops.add(t31, t12)
        del t31, t12
        
        # pd_op.reshape: (-1x7x7x192xf32) <- (-1x49x192xf32, 4xi64)
        t33 = paddle._C_ops.reshape(t32, t13)
        del t32, t13
        
        # pd_op.reshape: (-1x4x4x7x7x192xf32) <- (-1x7x7x192xf32, 6xi64)
        t34 = paddle._C_ops.reshape(t33, t14)
        del t14, t33
        
        # pd_op.transpose: (-1x4x7x4x7x192xf32) <- (-1x4x4x7x7x192xf32)
        t35 = paddle._C_ops.transpose(t34, [0, 1, 3, 2, 4, 5])
        del t34
        
        # pd_op.reshape: (-1x28x28x192xf32) <- (-1x4x7x4x7x192xf32, 4xi64)
        t36 = paddle._C_ops.reshape(t35, t15)
        del t15, t35
        
        return t36

    def get_input_spec(self):
        return [
            # t0
            paddle.static.InputSpec(shape=[], dtype='int64'),
            # t1
            paddle.static.InputSpec(shape=[], dtype='int64'),
            # t2
            paddle.static.InputSpec(shape=[], dtype='int64'),
            # t3
            paddle.static.InputSpec(shape=[], dtype='int64'),
            # t4
            paddle.static.InputSpec(shape=[None, 6, 49, 49], dtype='float32'),
            # t5
            paddle.static.InputSpec(shape=[16, 49, 49], dtype='float32'),
            # t6
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # t7
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # t8
            paddle.static.InputSpec(shape=[], dtype='int64'),
            # t9
            paddle.static.InputSpec(shape=[None, 6, 49, 32], dtype='float32'),
            # t10
            paddle.static.InputSpec(shape=[], dtype='int64'),
            # t11
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            # t12
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # t13
            paddle.static.InputSpec(shape=[4], dtype='int64'),
            # t14
            paddle.static.InputSpec(shape=[6], dtype='int64'),
            # t15
            paddle.static.InputSpec(shape=[4], dtype='int64'),
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
            paddle.to_tensor(128, dtype='int64').reshape([]),
            # t1
            paddle.to_tensor(16, dtype='int64').reshape([]),
            # t2
            paddle.to_tensor(6, dtype='int64').reshape([]),
            # t3
            paddle.to_tensor(49, dtype='int64').reshape([]),
            # t4
            paddle.uniform([2048, 6, 49, 49], dtype='float32', min=0, max=0.5),
            # t5
            paddle.uniform([16, 49, 49], dtype='float32', min=0, max=0.5),
            # t6
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # t7
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            # t8
            paddle.to_tensor(2048, dtype='int64').reshape([]),
            # t9
            paddle.uniform([2048, 6, 49, 32], dtype='float32', min=0, max=0.5),
            # t10
            paddle.to_tensor(192, dtype='int64').reshape([]),
            # t11
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            # t12
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # t13
            paddle.to_tensor([-1, 7, 7, 192], dtype='int64').reshape([4]),
            # t14
            paddle.to_tensor([-1, 4, 4, 7, 7, 192], dtype='int64').reshape([6]),
            # t15
            paddle.to_tensor([-1, 28, 28, 192], dtype='int64').reshape([4]),
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