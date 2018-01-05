## Profiling T3F
To profile the library, use the following commands
```bash
# Running on CPU.
CUDA_VISIBLE_DEVICES= python profile.py --file_path logs_cpu.pkl
# Running on GPU.
CUDA_VISIBLE_DEVICES=0 python profile.py --file_path logs_gpu.pkl
```
To visualize the results in a table, see ```results.ipynb``` Jupyter notebook.
Here are the numbers you can get on NVIDIA DGX-1 server with Tesla V100 GPU and Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz with 80 logical cores
 <img src="results.png" height="200">
#TODO what time units?

Note that by default TensorFlow has a very slow ```tf.transpose``` op for tensors of more than 5 dimensions, which affects the results a lot. To achieve the performance as described above, apply [the following patch](https://github.com/tensorflow/tensorflow/pull/15893) and [compile TensorFlow from sources](https://www.tensorflow.org/install/install_sources).

## Comparing against TTPY
To benchmark T3F against another library for Tensor Train decomposition [TTPY](github.com/oseledets/ttpy), install TTPY and run the following command in the bash shell
```bash
python profile_ttpy.py --file_path logs_ttpy.py
```
