import pytest

pytest.importorskip("torch", reason="PyTorch not installed. Please install with: pip install dace[ml]")
import torch
import torch.nn as nn
from dace.frontend.python.module import DaceModule
from tests.utils import torch_tensors_close
import copy


class MinimalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32, 128),
            nn.SiLU(),
            nn.Linear(128, 2)
        )

    def forward(self, input):
        x = self.flatten(input)
        output = self.linear_relu_stack(x)
        return output


# No parameter updates needed! We only care fr the correctness of the computed gradients
@pytest.mark.torch
@pytest.mark.autodiff
def test_jiayi_minimalMLP(sdfg_name):

    # Generate inputs
    input = torch.randn(100, 8, 4)

    # Create torch instance and DaCe module
    torch.manual_seed(0)
    torch_mlp = MinimalMLP()
    dace_mlp = DaceModule(copy.deepcopy(torch_mlp),
                        sdfg_name=sdfg_name,
                        onnx_simplify=True,
                        simplify=False,
                        backward=True)

    input_torch = torch.clone(input)
    input_torch.requires_grad = True

    input_dace = torch.clone(input)
    input_dace.requires_grad = True

    output_torch = torch_mlp(input_torch)
    output_dace = dace_mlp(input_dace)
    
    # print(type(output_torch), type(output_dace))
    # print(output_torch)
    # print(output_dace)

    # # Compare outputs
    # assert torch_tensors_close("outputs of MLP", output_torch, output_dace, rtol=1e-3, atol=1e-3), "Outputs do not match!"

    # Forward on both with a dummy sum(y_i) loss function
    output_torch.sum().backward()
    output_dace.sum().backward()

    # Check gradients of the parameters
    for (name, dace_param), (pt_name, pt_param) in zip(torch_mlp.named_parameters(), dace_mlp.named_parameters()):
        assert 'model.' + name == pt_name, f"Parameter name mismatch: expected 'model.{name}', got '{pt_name}'"
        torch_tensors_close(name, dace_param.grad, pt_param.grad)

    # Check the gradients of the input tensor
    torch_tensors_close("input_grad", input_torch.grad, input_dace.grad)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
