set -x
OF_CFLAGS=( $(python3 -c 'import oneflow; print(" ".join(oneflow.sysconfig.get_compile_flags()))') )
OF_LFLAGS=( $(python3 -c 'import oneflow; print(" ".join(oneflow.sysconfig.get_link_flags()))') )

# nvcc -std=c++11 -c -o roi_align_kernel.cu.o roi_align_kernel.cu ${OF_CFLAGS[@]}  -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc -I/usr/include/python3.6m -c ops/roi_align.cu -o build/temp.linux-x86_64-3.6/ops/roi_align.o ${OF_LFLAGS[@]} ${OF_CFLAGS[@]} -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_GLIBCXX_USE_CXX11_ABI=0 -DWITH_CUDA -O2 --cudart=static --disable-warnings --compiler-options -fPIC -ccbin=g++ -std=c++11 -dc
nvcc -dlink -o build/temp.linux-x86_64-3.6/ops/roi_align_link.o \
    build/temp.linux-x86_64-3.6/ops/roi_align.o \
    /home/caishenghang/oneflow/build/CMakeFiles/of_ccobj.dir/oneflow/core/kernel/of_ccobj_generated_kernel_util.cu.o \
    /home/caishenghang/oneflow/build/libof_ccobj.a \
    -lcudadevrt -lcudart ${OF_LFLAGS[@]} ${OF_CFLAGS[@]}  -Xlinker -fPIC
g++ -pthread -shared -Wall -Wl,-z,relro -g build/temp.linux-x86_64-3.6/ops/roi_align_link.o -L/usr/lib64 -lpython3.6m -o build/lib.linux-x86_64-3.6/oneflow_detection/lib.cpython-36m-x86_64-linux-gnu.so -lcudart -lcudadevrt -L/usr/local/cuda/lib64 -x cu -L/home/caishenghang/oneflow/build/python_scripts/oneflow -l:_oneflow_internal.so
# python3 test/roi_align_test.py 
