# Use the system CUDA
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH

# Headers (so it finds nv/target, etc.)
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH

# Libraries: make sure ld can find libcuda.so stub
export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64/stubs:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Then reinstall
pip install --no-build-isolation --force-reinstall sageattention==2.2.0