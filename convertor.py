# MIT Licence
# Yash Bonde

import os
import sys
import subprocess
from argparse import ArgumentParser

from transformers import AutoModel

def get_model(args):
  # download model
  # export to XML
  pass


def openvino_optimize(path_to_xml, args):
  pass

if __name__ == "__main__":
  args = ArgumentParser()
  args.add_argument("--model", default = None, help = "huggingface model name", required = True)
  args.add_argument("--ov_folder", default=None, help="Path to OpenVino folder", required=True)
  args = args.parse_args()





