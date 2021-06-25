from datasets import build_dataset
from main import get_args_parser
from timm.models import create_model
import os
import json
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.utils.bundled_inputs
import xcit
import shlex
from pathlib import Path

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

CHECKPOINT_PATH = "xcit_tiny_12_p16_224_dist.pth"
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

# parser = get_args_parser()
# args = parser.parse_args()
# args.pretrained = "xcit_tiny_12_p16_224_dist.pth"
# args.model = "xcit_tiny_12_p16"
args = checkpoint['args']

def print_diff_args(args, parser):
    default_args = parser.parse_args([])
    set_args = ["python", "run_with_submitit.py", "--partition", "learnlab"]
    for k, default_value in vars(default_args).items():
        try:
            v = getattr(args, k)
        except AttributeError:
            continue
        if k == "world_size":
            assert v <= 8
            set_args += ["--nodes", str(1), "--ngpus", str(v)]
            continue
        if k in ["dist_url", "resume", "output_dir"]:
            continue
        if default_value != v:
            arg_name = f"--{k}"
            try:
                parser.parse_args([arg_name, str(v)])
            except:
                arg_name = arg_name.replace("_", "-")
            set_args += [arg_name, str(v)]
    print(shlex.join(set_args))
    import ipdb; ipdb.set_trace()

args.model = "xcit_tiny_12_p16"
print(args)
print_diff_args(args, get_args_parser())

dataset, _ = build_dataset(is_train=False, args=args)


model = create_model(
    args.model,
    pretrained=False,
    num_classes=args.nb_classes,
    drop_rate=args.drop,
    drop_path_rate=args.drop_path,
    drop_block_rate=None
)
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()

# TODO: Fuse Conv, bn and relu
def auto_fuse_inplace(net) -> None:
    """
    Can only fuse:
    conv, bn
    conv, bn, relu
    conv, relu
    linear, relu
    """
    if isinstance(net, torch.nn.Sequential):
        if len(net) == 2 and isinstance(net[0], torch.nn.Conv2d) and isinstance(net[1], torch.nn.BatchNorm2d):
            torch.quantization.fuse_modules(net, ['0', '1'], inplace=True)

    if hasattr(net, 'fuse_model'):
        net.fuse_model()

    for name, child in net.named_children():
        auto_fuse_inplace(child)
auto_fuse_inplace(model)


## Model and data ready, let's quantize
class ModelWrapper(nn.Module):
    """ Wrapper around the model to allow regular inputs (rather than quantized ones) """
    def __init__(self, module) -> None:
        super().__init__()
        self._module = module
        self._quant = torch.quantization.QuantStub()
        self._dequant = torch.quantization.DeQuantStub()
    
    def forward(self, image) -> torch.Tensor:
        return self._dequant(self._module(self._quant(image)))

model_ = model
model = ModelWrapper(model)
model.eval()
print("Size of model before quantization")
print_size_of_model(model)

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
# model.qconfig = torch.quantization.default_qconfig
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
print(model.qconfig)

# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')
torch.quantization.prepare(model, inplace=True)

for i in range(10):
    image, _label = dataset[i]
    model(image.unsqueeze(0))

print('\n Observers AFTER \n\n', model_.blocks[0])
print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.quantization.convert(model, inplace=True)
print('Post Training Quantization: Convert done')

print("Size of model after quantization")
print_size_of_model(model)

# Now let's convert to JIT
sample_input = dataset[0][0].unsqueeze(0)
model_traced = torch.jit.trace(model, sample_input)
model_traced(sample_input)
optimize_for_mobile(model_traced, backend="CPU")
torch.utils.bundled_inputs.augment_model_with_bundled_inputs(model_traced, [(torch.zeros_like(sample_input), )])
extra_files = {
    "info": json.dumps({
        "input_shape": [list(sample_input.shape)],
        "from_checkpoint": CHECKPOINT_PATH,
        "code": Path(__file__).read_text(),
    })
}
model_traced.save(f"{CHECKPOINT_PATH}.jit", _extra_files=extra_files)
model_traced._save_for_lite_interpreter(f"{CHECKPOINT_PATH}_3.ptl", _extra_files=extra_files)