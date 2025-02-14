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

    def forward(self, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14):
        # pd_op.full: (1xi32) <- ()
        t15 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())
        
        # pd_op.assign: (1xi32) <- (1xi32)
        t16 = t15
        
        # pd_op.assign: (1xi32) <- (1xi32)
        t17 = t15
        
        # builtin.combine: ([512x4xf32, 512x4xf32]) <- (512x4xf32, 512x4xf32)
        t18 = [t0, t1]
        del t0, t1
        
        # pd_op.concat: (1024x4xf32) <- ([512x4xf32, 512x4xf32], 1xi32)
        t19 = paddle._C_ops.concat(t18, t15)
        del t18
        
        # pd_op.distribute_fpn_proposals: ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], [-1xi32, -1xi32, -1xi32, -1xi32], -1x1xi32) <- (1024x4xf32, 2xi64)
        t20, t21, t22 = (lambda x, f: f(x))(paddle._C_ops.distribute_fpn_proposals(t19, t2, 2, 5, 4, 224, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))
        del t19, t2
        
        # builtin.split: (-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32) <- ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32])
        t23, t24, t25, t26, = t20
        del t20
        
        # builtin.split: (-1xi32, -1xi32, -1xi32, -1xi32) <- ([-1xi32, -1xi32, -1xi32, -1xi32])
        t27, t28, t29, t30, = t21
        del t21
        
        # pd_op.roi_align: (-1x256x7x7xf32) <- (2x256x240x176xf32, -1x4xf32, -1xi32)
        t31 = paddle._C_ops.roi_align(t3, t23, t27, 7, 7, float('0.25'), 0, True)
        del t3
        
        # pd_op.roi_align: (-1x256x7x7xf32) <- (2x256x120x88xf32, -1x4xf32, -1xi32)
        t32 = paddle._C_ops.roi_align(t4, t24, t28, 7, 7, float('0.125'), 0, True)
        del t4
        
        # pd_op.roi_align: (-1x256x7x7xf32) <- (2x256x60x44xf32, -1x4xf32, -1xi32)
        t33 = paddle._C_ops.roi_align(t5, t25, t29, 7, 7, float('0.0625'), 0, True)
        del t5
        
        # pd_op.roi_align: (-1x256x7x7xf32) <- (2x256x30x22xf32, -1x4xf32, -1xi32)
        t34 = paddle._C_ops.roi_align(t6, t26, t30, 7, 7, float('0.03125'), 0, True)
        del t6
        
        # builtin.combine: ([-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32]) <- (-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32)
        t35 = [t31, t32, t33, t34]
        
        # pd_op.concat: (-1x256x7x7xf32) <- ([-1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32, -1x256x7x7xf32], 1xi32)
        t36 = paddle._C_ops.concat(t35, t15)
        del t35
        
        # pd_op.gather: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x1xi32, 1xi32)
        t37 = paddle._C_ops.gather(t36, t22, t15)
        del t15
        
        # pd_op.flatten: (-1x12544xf32) <- (-1x256x7x7xf32)
        t38 = paddle._C_ops.flatten(t37, 1, 3)
        
        # pd_op.matmul: (-1x1024xf32) <- (-1x12544xf32, 12544x1024xf32)
        t39 = paddle._C_ops.matmul(t38, t7, False, False)
        del t7
        
        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        t40 = paddle._C_ops.add(t39, t8)
        del t8
        
        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        t41 = paddle._C_ops.relu(t40)
        del t40
        
        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        t42 = paddle._C_ops.matmul(t41, t9, False, False)
        del t9
        
        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        t43 = paddle._C_ops.add(t42, t10)
        del t10
        
        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        t44 = paddle._C_ops.relu(t43)
        del t43
        
        # pd_op.matmul: (-1x5xf32) <- (-1x1024xf32, 1024x5xf32)
        t45 = paddle._C_ops.matmul(t44, t11, False, False)
        del t11
        
        # pd_op.add: (-1x5xf32) <- (-1x5xf32, 5xf32)
        t46 = paddle._C_ops.add(t45, t12)
        del t12
        
        # pd_op.matmul: (-1x16xf32) <- (-1x1024xf32, 1024x16xf32)
        t47 = paddle._C_ops.matmul(t44, t13, False, False)
        del t13
        
        # pd_op.add: (-1x16xf32) <- (-1x16xf32, 16xf32)
        t48 = paddle._C_ops.add(t47, t14)
        del t14
        
        return t16, t17, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t36, t37, t38, t39, t41, t42, t44, t45, t46, t47, t48

    def get_input_spec(self):
        return [
            # t0
            paddle.static.InputSpec(shape=[512, 4], dtype='float32'),
            # t1
            paddle.static.InputSpec(shape=[512, 4], dtype='float32'),
            # t2
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            # t3
            paddle.static.InputSpec(shape=[2, 256, 240, 176], dtype='float32'),
            # t4
            paddle.static.InputSpec(shape=[2, 256, 120, 88], dtype='float32'),
            # t5
            paddle.static.InputSpec(shape=[2, 256, 60, 44], dtype='float32'),
            # t6
            paddle.static.InputSpec(shape=[2, 256, 30, 22], dtype='float32'),
            # t7
            paddle.static.InputSpec(shape=[12544, 1024], dtype='float32'),
            # t8
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # t9
            paddle.static.InputSpec(shape=[1024, 1024], dtype='float32'),
            # t10
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # t11
            paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
            # t12
            paddle.static.InputSpec(shape=[5], dtype='float32'),
            # t13
            paddle.static.InputSpec(shape=[1024, 16], dtype='float32'),
            # t14
            paddle.static.InputSpec(shape=[16], dtype='float32'),
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
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            # t1
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            # t2
            paddle.to_tensor([512, 512], dtype='int64').reshape([2]),
            # t3
            paddle.uniform([2, 256, 240, 176], dtype='float32', min=0, max=0.5),
            # t4
            paddle.uniform([2, 256, 120, 88], dtype='float32', min=0, max=0.5),
            # t5
            paddle.uniform([2, 256, 60, 44], dtype='float32', min=0, max=0.5),
            # t6
            paddle.uniform([2, 256, 30, 22], dtype='float32', min=0, max=0.5),
            # t7
            paddle.uniform([12544, 1024], dtype='float32', min=0, max=0.5),
            # t8
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # t9
            paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
            # t10
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # t11
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            # t12
            paddle.to_tensor([0.19392463564872742, 0.030198199674487114, 0.15864147245883942, 0.1595998853445053, 0.40606245398521423], dtype='float32').reshape([5]),
            # t13
            paddle.uniform([1024, 16], dtype='float32', min=0, max=0.5),
            # t14
            paddle.to_tensor([0.1932821273803711, 0.3028981685638428, 0.18616729974746704, 0.04886952415108681, 0.30841556191444397, 0.4989411532878876, 0.3225594460964203, 0.14801165461540222, 0.494112491607666, 0.2424657791852951, 0.23623594641685486, 0.49401646852493286, 0.31682002544403076, 0.2843758463859558, 0.2656653821468353, 0.39931434392929077], dtype='float32').reshape([16]),
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