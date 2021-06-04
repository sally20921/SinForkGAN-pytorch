# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch

ONNX_EXPORT = False


def model_info(model):
    # Plots a line-by-line description of a PyTorch model
    parameter_num = sum(x.numel() for x in model.parameters())
    gradient_num = sum(x.numel() for x in model.parameters() if x.requires_grad)

    try:
        from thop import profile
        macs, _ = profile(model, inputs=(torch.zeros(1, 3, 640, 640),))
        FLOPs = f', {macs / 1E9 * 2:.1f} GFLOPS'
    except:
        FLOPs = ''

    print(f"Model Summary: {len(list(model.parameters()))} layers, "
          f"{parameter_num} parameters, {gradient_num} gradients{FLOPs}")
