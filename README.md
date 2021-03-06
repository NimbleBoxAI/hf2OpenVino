# hf2OpenVino
Scripts to optimize any huggingface model. Before running the script you will have to modify the hugginface code to avoid `TracerWarnings`. Once this script is run you will see a tracer warning, you will have to manually delete the problematic piece.
```
python3 converter.py --name=gpt2 \
--auto=AutoModelWithLMHead \
--ov_folder=../openvino/openvino_2021 \
--size=[1, 768] \
--export_ov=./gpt2export \
--export_onnx=./gpt2export/gpt2.onnx
```

For example the above code throws error in `modeling_gpt2.py` file:
```python
if not self.is_cross_attention:
  # if only "normal" attention layer implements causal mask
  mask = self.bias[:, :, ns - nd : ns, :ns]
  w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))
```

This part would need to be commented out.


## Installation

Running this is a bit nasty so here is the tutorial on how to get it work. Assume that we are converting the `gpt2` model from hf.

If you have python-3.9 installed you will have to install python-3.7.9 for this project to work. Follow the instructions below when building for first time (verified build on MacOS):
```
brew install pyenv                             # for syncing multitple versions on the machine
pip3 install virtualenv                        # virtual-environment maker, can use any other package
pyenv install 3.7.9                            # install the specific version
pyenv local 3.7.9                              # set local (this folder) version to 3.7.9
export LOCAL_PY_VER_PATH=`pyenv which python3` # set path for convinience
echo $LOCAL_PY_VER_PATH                        # [opt.] to check the path
$LOCAL_PY_VER_PATH -m venv .                   # using the path above build a virtual environment in this folder
source bin/activate                            # activate the local env
pip3 install -r requirements.txt               # install run dependencies
```

When coming back to this project simply activate the virtualenv as and the rest will be ready for you:
```
source bin/activate
```

## ONNX Model

To get the model in the ONNX format first run the file `convert.py`, this should dump `gpt2.onnx` file.
```
python3 convert.py
```

## From ONNX to Openvino

For this you must first have openvino installed on your system. Download from [here](https://software.intel.com/en-us/openvino-toolkit). Now I have added most of the requirements in my `requirements.txt` file, however you should also install those for OpenVino. After that run the following commands to setup environment variables:
```
export OPENVINO_FOLDER="path/to/openvino_2021"
cd $OPENVINO_FOLDER/bin
source setupvars.sh
cd $OPENVINO_FOLDER/deployment_tools/model_optimizer
pip3 install install -r requirements.txt
pip3 install install -r requirements_onnx.txt
```

If everything works correctly you will see an output like this:
```
[setupvars.sh] OpenVINO environment initialized
```

Now come back to this repo, Openvino environment setup works correctly only if you are in the `openvino_2021/bin` folder. Now we run the script `mo_onnx.py`:
```
mo_onnx.py --help                              # to get meanings of arguments to be passed
mkdir full_precision half_precision            # full_precision is FP36 and other is FP16
mo_onnx.py --input_model gpt2.onnx \
--data_type=FP32/FP16 \
--output_dir=full_precision/half_precision
```

If everything works correctly you should see 3 files in `/fp32` folder:
```
gpt2.bin
gpt2.mapping
gpt2.xml
```