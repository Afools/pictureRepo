# CUDA 基础

## 核函数

cuda 通过调用核函数来执行并行计算，核函数用__global__声明，调用时需要使用<<<grid_size,block_size>>>指定调用的线程数量。

grid_size 指定调用的块的数量  
block_size 指定块中线程的数量 

核函数中这两个参数分别存储在内置预定义变量gridDim.x 和 blockDim.x中.    
gridDim规定blockIdx的范围,blockDim规定threadIdx的范围  

``` cpp
int idx = blockIdx.x*gridDim.x+threadIdx.x    // 这样定位到执行的线程
int stride = blockDim.x * gridDim.x           // 指定步长
```

## 并行计算
``` cpp
 cudaError_t cudaMalloc(void** devPtr,sizet size);
```
在device申请显存
> 注意，cuda中分配内存的方式不只cudaMalloc，cudaMalloc分配的是GPU内存，需要在CPU和GPU上手动使用`cudaMemcpy()`进行数据迁移  
> 而`cudaMallocManaged` 可以分配统一内存（即Unified Memory），无需手动释放内存和迁移数据。
```cpp
cudaError_t cudaMemcpy(void* dst, const void* src, sizet count, cudaMemcpyKind kind);
```
在host和device之间进行数据通信。 src指向源数据， dst是目标区域， count是复制的字节数，kind给出复制的方向： 如cudaMemcpyHostToDevice是讲host数据拷贝到device上。
*注意* cudaMemcpy是阻塞的，在完成之前CPU无法处理后续任务。

## GPU 并行结构
GPU并行计算从逻辑上分为grid，block，thread三层，从硬件上分为multiprocessor，wrap，thread三层。  

### 逻辑层面并行
SM（Stream Multiprocessor）是最大的处理单位，核函数启动后，每个block会被同时分配给可用的SM上（之后不会重新分配）。每个block中的thread在SM上并行执行。

### 硬件层面并行
硬件层面上将线程聚合层线程束wrap，32个线程组成一个wrap，在机器层面每个SM只执行一个线程束，线程束的每个线程执行同一条机器指令，包括有分支的部分。  
在面临分支时，未进入分支的线程需要等待其他线程执行完，然后继续。  
在Fermi架构中，当一个block分配给一个SM时，block会拆分成多个wrap，在SM上交替进行，SM上的线程束切换没有开销。
### 同步
显式的希望主机等待设备端执行可以使用：
```cpp
cudaError_t cudaDeviceSynchronize(void);
```
也可以使用隐式的方法让主机无法在设备端执行完之前继续进行，比如内存拷贝函数：
```cpp
cudaError_t cudaMemcpy(void* dst, const void *src, size_t count, cudaMemcpyKind kind)；
```
在核函数启动后下一条指令就是从设备复制数据到主机，那么主机必须等待设备就绪。

## 性能评估

### 时间性能
不能用clock()进行计时，因为clock()根据cpu时钟数和cpu每秒时钟数（常量）进行计算得到最终时间，在多线程并行计算中会将每个线程的时钟数都记录在内，最终时间会大于实际时间。  
因此我们使用gettimeofday()函数：
```cpp
#include <sys/time.h>
double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
}
```

# 环境配置问题

### 编译出错，cuda内置变量及函数无法识别
文件名应为.cu后缀而不能用.cpp后缀。使用.cu后缀并用nvcc编译时可以不添加cuda.h头。

> 疑问： 如果不添加cuda头文件，如何判定调用的是运行时api还是驱动api