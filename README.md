# dancing-plant

Tools for tracing leaf an stem motion from RGB video.

The typical pipeline for this process is:
1) Generate dense UV flow maps using RAFT
2) Generate traces over time starting from a subsampled 2D mesh of pixels
3) Extract fastest moving traces to remove those generated from lighting and other noise
4) Cluster traces using dynamic time warping
5) Save and display results, which can be post-processed (e.g. sonification)

A high-level user interface for using this package is provided in the /launch directory.

**See the in-code documentation provided in each launch file.**

| Python Launch File | Description |
| ------------------ | ----------- |
| gen_flow.py | generate UV flow maps with a pretrained RAFT model |
| gen_trace.py | generate trace CSVs and complementary images using UV flow maps |
| cluster_trace.py | cluster traces from gen_trace.py using dynamic time warping |
| draw_trace.py | annotate video frames with traces from gen_trace.py |
| imsort.py | access file name sorting and indexing tools |

**clean_up.sh** is also provided to conveniently erase all existing images and CSVs generated.

All files in dancing_plant/raft, dancing_plant/flow.py, and alt_cuda_corr/, along with the model downloaded via **download_raft_model.sh**, are directly adapted from the [RAFT](https://github.com/princeton-vl/RAFT) GitHub repository.

## Installation

This project has been tested using Python 3.7 on Ubuntu 16.04.

1) Change directories to this project's root and install the dancing-plant package.

```
pip install -e .
```

2) Install PyTorch with CUDA support. PyTorch version 1.6.0 and CudaToolKit 10.1 are tested to work with the following step.

3) Compile RAFT's custom CUDA extension which significantly reduces GPU memory requirements.

```
cd alt_cuda_corr && python setup.py install && cd ..
```

4) Install other Python requirements.

```
pip install -r requirements.txt
(Arc: pip install -U future if the future package is missing)
```

5) Download the RAFT model.

```
./download_raft_model.sh
```

cmd2 0.8.9 requires pyperclip, which is not installed.
cmd2 0.8.9 requires wcwidth; sys_platform != "win32", which is not installed.
torch 1.6.0 requires future, which is not installed.

Arc Instruction:
infer1.arc.vt.edu
Firs-time use:
module load EasyBuild
<!-- module load python37 -->
<!-- module load CUDA/11.1.1-GCC-10.2.0 -->
module load CUDA/10.1.243-GCC-8.3.0
module load Anaconda3/2020.11
nvidia-smi  ## make sure you see GPUs
module save
conda create -n pytorch python=3.7.0
unset PYTHONPATH
source activate pytorch
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch 
export CXX=g++ 
git clone https://github.com/alanwangvt/dancing-plant.git
"run steps 1,3,4,5 above"
use "pip install" to install any missing packages

<!-- In ~/.bashrc, add: alias python='/cm/local/apps/python37/bin/python'
                    export PYTHONPATH=/home/alanwang/dancing-plant:$PYTHONPATH
                    export CXX=g++ 
            then: source ~/.bashrc -->

After first-time:
module restore
unset PYTHONPATH
source activate pytorch