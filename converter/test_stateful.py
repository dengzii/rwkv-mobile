import numpy as np
import torch

import coremltools as ct

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("accumulator", torch.tensor(np.array([0] * 1024, dtype=np.float16)))
        self.linear = torch.nn.Linear(1024, 1024).half()

    def forward(self, x):
        self.accumulator += x
        return self.linear(self.accumulator) * self.accumulator

traced_model = torch.jit.trace(Model().eval(), torch.tensor([1] * 1024))
mlmodel = ct.convert(
    traced_model,
    inputs = [ ct.TensorType(shape=(1024,)) ],
    outputs = [ ct.TensorType(name="y") ],
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1024,),
            ),
            name="accumulator",
        ),
    ],
    minimum_deployment_target=ct.target.iOS18,
)

mlmodel.save("test_stateful.mlpackage")
del mlmodel

model = ct.models.MLModel("test_stateful.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
state = model.make_state()

inputs = {'x': np.array([1.0] * 1024)}
for i in range(10):
    outputs = model.predict(inputs, state=state)
    print(outputs['y'])
