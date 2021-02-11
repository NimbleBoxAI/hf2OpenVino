# MIT Licence
# Yash Bonde

import os
import sys
import torch
import subprocess
from argparse import ArgumentParser

import transformers
from transformers import AutoModel

AUTO_HEAD_MAPPING = {
  x: getattr(transformers.models.auto, x)
  for x in list(set([
    str(x) for x in dir(transformers.models.auto) if "AutoModel" in x
  ]))
}

def get_model(args):
  # download model and export to onnx
  print("-"*70+"\nDownloading Model")
  model = AUTO_HEAD_MAPPING[args.auto].from_pretrained(args.name)
  input_ = torch.randint(high = args.random_high, size = args.size).long()
  print(f"Exporting Model at {args.export_onnx}")
  torch.onnx.export(model, input_, args.export_onnx)

def openvino_optimize(args):
  print("-"*70 + "Setting up environment variables")
  print("source", f"{args.ov_folder}/bin/setupvars.sh")
  subprocess.run([
      "sh", f"{args.ov_folder}/bin/setupvars.sh"
  ])

  req = f"{args.ov_folder}/deployment_tools/model_optimizer/requirements.txt"
  req_onnx = f"{args.ov_folder}/deployment_tools/model_optimizer/requirements_onnx.txt"
  subprocess.run(["pip3", "install", "-r", req])
  subprocess.run(["pip3", "install", "-r", req_onnx])

  print(f"Starting openvino optimisation and exporting to folder {args.export_ov}")
  mo_path = f"{args.ov_folder}/deployment_tools/model_optimizer/mo_onnx.py"
  subprocess.run([
    "python3",  mo_path, "--input_model", args.export_onnx,
    "--output_dir", args.export_ov
  ])


if __name__ == "__main__":
  args = ArgumentParser()
  args.add_argument(
    "--name",
    type = str,
    help = "huggingface model name",
    required = True
  )
  args.add_argument(
    "--export_onnx",
    type=str,
    help="name of the output file with .onnx extension",
    required=True
  )
  args.add_argument(
      "--export_ov",
      type=str,
      help="folder where openvino model will be put",
      required=True
  )
  # we need to know which model head does the user want
  args.add_argument(
    "--auto",
    type = str,
    help = "name of the AutoXXX to call",
    choices=list(AUTO_HEAD_MAPPING.keys()),
    required=True
  )
  args.add_argument(
    "--random_high",
    type=int,
    default = 100,
    help="maximum value of input to give to model (vocabulary size)",
  ) # most models will have vocab size of atleast 100
  args.add_argument(
    "--size",
    type=str,
    help="input size for ONNX conversion, once this is fixed at runtime you will "\
      "always have to give it the same size. Numbers seperated by comma with no spaces",
    required=True
  )
  args.add_argument("--ov_folder", default=None, help="Path to OpenVino folder", required=True)
  args = args.parse_args()

  args.export_onnx = os.path.abspath(args.export_onnx) # expand to full path
  args.ov_folder = os.path.abspath(args.ov_folder)     # expand to full path
  args.size = [int(x) for x in args.size.split(",")]    # convert string to shape

  os.makedirs(args.export_ov, exist_ok = True)
  get_model(args)
  openvino_optimize(args)
