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
在面临分支时，未进入分支的线程需要等待其他线程执行完，然后继续。这种线程束分化方法在面临多分枝时会造成严重的性能浪费。  
在Fermi架构中，当一个block分配给一个SM时，block会拆分成多个wrap，在SM上交替进行，SM上的线程束切换没有开销。

### 动态并行
动态并行指的是核函数中的线程启动了另一个核函数，这样便分出了父核函数和子核函数，父线程、父网格、父线程块，子线程、子网格、子线程块。动态并行相当于串行编程中的递归，可以让复杂的内核更有层次，可以动态的利用GPU硬件调度器和加载平衡器，并且在内核中启动内核可以减少一部分传输消耗。但是动态并行会让程序编写更加复杂和难以控制。  

父线程块会等待所有子线程执行完毕才能退出。
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

----------  
可以使用`__syncthread();`函数对同一个块内的线程进行同步，线程会同时停止在某个设定的位置。这个函数只能同步同一个块内的线程，不能同步不同块内的线程，想要同步不同块内的线程，就只能让核函数执行完成，控制程序交换主机，这种方式来同步所有线程。

## 内存管理

### CUDA 内存模型
相比于CPU的可编程内存，CUDA的内存模型提供了更丰富的可控制内存设备，如下：
- 寄存器
- 共享内存
- 本地内存
- 常量内存
- 纹理内存
- 全局内存
CUDA中每个线程都有私有的本地内存；线程块有自己的共享内存，对线程块的所有线程可见；所有线程都能访问读取常量内存和纹理内存，但是不能写；全局内存、常量内存和纹理内存有相同的生命周期，有不同的用途。
![](https://face2ai.com/CUDA-F-4-1-%E5%86%85%E5%AD%98%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B0/1-5.png)

### 寄存器
寄存器是每个线程私有的变量存储空间，在核函数中不加修饰的声明一个变量时，此变量就存储在寄存器中，包括核函数中定义的常量长度的数组。寄存器溢出会导致本地内存来提供额外空间，这是要尽力避免的。
```cpp
__global__ void
__lauch_bounds__(maxThreadaPerBlock,minBlocksPerMultiprocessor)
kernel(...) {
    /* kernel code */
}
```
这里面在核函数定义前加了一个 关键字 lauch_bounds，然后他后面对应了两个变量：

- maxThreadaPerBlock：线程块内包含的最大线程数，线程块由核函数来启动
- minBlocksPerMultiprocessor：可选参数，每个SM中预期的最小的常驻内存块参数。  
注意，对于一定的核函数，优化的启动边界会因为不同的结构而不同。也可以在编译选项中加入`-maxrregcount=32`来控制一个编译单元里所有核函数使用的最大数量。

### 本地内存

核函数中所有符合进入寄存器条件但不能进入寄存器的数据被存放在本地内存中。不满足核函数寄存器限定条件的变量也放入本地内存中。本地内存通常存储在一级或二级缓存中。

### 共享内存
线程块内的线程共享一定数量的SM内存，使用时需要用`__share__`进行修饰。  
当线程块需要的共享内存过大会导致SM中并行的线程块减少，影响活跃线程束的数量。  
共享内存块内线程可见，可能导致内存竞争，也可以通过共享内存通信。可以使用同步语句`void __syncthread();`避免内存竞争，此语句需要所有线程都运行到这一同步点后才能进行下一步，可以利用这点设计出避免内存竞争的程序。

### 常量内存
常量内存存放在设备内存中，每个SM都有专用的常量内存缓存，使用时用`__constant__`进行修饰。  
常量内存读取时会直接广播给线程束内的所有线程，因此在所有线程读取相同地址的场景效率很高。  
常量内存无法被核函数修改，但是可以通过主机端初始化，函数如下：
```cpp
cudaError_t cudaMemcpyToSymbol(const void* symbol, const void *src, size_t count);  //从src复制count个字节内存到symbol里。
```


### 纹理内存
纹理是只读的片上缓存区域，用于加速对图像或多维数据的随机访问。能够硬件支持访问时插值计算。

### 全局内存
GPU上最大的存储空间，global指的是生命周期和作用域，一般在主机端定义，也可以在设备端定义，和应用程序同生命周期。可以用`__device__`在设备代码中静态声明一个变量，也可以动态声明。

### 全局静态内存
全局静态内存用`__device__`修饰，并用`cudaMemcpyToSymbol()`进行赋值，如下例：
```cpp
__device__ float devData;
__global__ void checkGlobal(){
  devData+=2.0;
}
int main(){
  float hostData=2.0f;
  cudaMemcpyToSymbol(devData,&hostData,sizeof(float));
  checkGlobal<<<1,1>>>();
  chdaMemcpyFromSymbol(&hostData,devData,sizeof(float));
  cudaDeviceReset();
  return 0;
}
```
不能用`cudaMemcpy()`对静态变量进行赋值，因为主机端不能直接对设备变量进行取地址操作。

### 全局内存访问
粒度是GPU读取内存的最小单位，即使需要的数据不足粒度大小也要读取一粒度大小的数据，因此在读取内存时要尽可能保证数据量是粒度的整数倍来减少浪费。  
数据起始地址也应该是粒度的整数倍（有些地方写的是偶数倍），可以减少跨界数据来减少带宽浪费。这种方法称为**对齐内存访问**。当一个线程束访问的内存都在一个内存块里，就会出现**合并访问**。  
**对齐合并访问能够更高效的进行内存读取。**

在启用L1缓存时，事务粒度通常为128；不启用L1缓存时，事务粒度通常为32。细粒度能提高带宽利用率。

### 全局内存写入
全局内存写入时通常不使用一级缓存（在某些架构上通过编译指令`-Xptxas -dlcm=cg`等方式强制让写入经过L1缓存）,存储操作粒度为32字节，将事务分为一段、两段、四段进行操作。根据内存存储需要和对应段地址是否连续进行分配。具体可查看[链接](https://face2ai.com/CUDA-F-4-3-%E5%86%85%E5%AD%98%E8%AE%BF%E9%97%AE%E6%A8%A1%E5%BC%8F/#:~:text=%E7%BC%93%E5%AD%98%E4%BA%86%E3%80%82-,%E5%85%A8%E5%B1%80%E5%86%85%E5%AD%98%E5%86%99%E5%85%A5,-%E5%86%85%E5%AD%98%E7%9A%84%E5%86%99)。

### 数组访问
CUDA对细粒度数组可以进行连续访问，访问效率较高；但是对于粗粒度的结构体数组(Array of Structure AoS)，想要访问其中的成员时访问就是不连续的，会降低访问效率。

### 访问优化方法
1. 通过增加SM上运行的wrap数量隐藏内存存取延迟，具体可以用展开循环、控制分支来实现
2. 通过尽量增加合并访问和对齐访问减少带宽浪费
3. 合理应用L1缓存减少全局内存访问次数，增加访问效率
4. 尽可能均匀使用DRAM来减少重复访问同一内存块的等待时间
   
## 共享内存和常量内存
### 共享内存概述
共享内存是在他所属的线程块被执行时建立，线程块执行完毕后共享内存释放，线程块和他的共享内存有相同的生命周期。  
**使用共享内存最重要的是避免线程的访问冲突。** 

声明共享内存通过关键字：
```cpp
__shared__
```
声明一个二维浮点数共享内存数组的方法是：
```cpp
__shared__ float a[size_x][size_y];
```
这里的size_x,size_y和声明c++数组一样，要是一个编译时确定的数字，不能是变量。
如果想动态声明一个共享内存数组，可以使用extern关键字，并在核函数启动时添加第三个参数。
声明:
```cpp
extern __shared__ int tile[];
```
在执行上面这个声明的核函数时，使用下面这种配置：
```cpp
kernel<<<grid,block,isize*sizeof(int)>>>(...);
```
isize就是共享内存要存储的数组的大小。比如一个十个元素的int数组，isize就是10.
注意，动态声明只支持一维数组。

## 共享内存存储体
存储体是共享内存的硬件实现形式，每个共享内存单元包含32个存储体，对应wrap的32个线程，可以同时访问。当每个线程访问不同的存储体时，只要一个事务就能完成；但如果出现访问冲突，就需要多个事务。  

并不是多个线程对同一个存储体访问就一定会出现访问冲突，因为存在多个线程访问一个地址的可能。如果多个线程访问同一存储体的同一地址（同一数据）会进行广播机制，即单一线程得到数据后向其他线程广播数据。这种读取方法延迟低但是利用率也低。  

存储体的宽度也会影响内存访问冲突的可能性。宽度是指存储体一次读取可以取出的数据大小，2.x计算能力设备为4字节，3.x为8字节。当线程读取时，如果宽度为8，两个线程分别需要读取连续8字节数据的左边4字节数据和右边4字节数据，则不会发生冲突（这个8字节数据起始地址是8字节的整数倍）。具体冲突产生方式可查看[链接](https://face2ai.com/CUDA-F-5-1-CUDA%E5%85%B1%E4%BA%AB%E5%86%85%E5%AD%98%E6%A6%82%E8%BF%B0/#:~:text=%E5%86%B2%E7%AA%81%EF%BC%8C%E6%8F%90%E9%AB%98%E6%80%A7%E8%83%BD%E3%80%82-,%E8%AE%BF%E9%97%AE%E6%A8%A1%E5%BC%8F,-%E5%85%B1%E4%BA%AB%E5%86%85%E5%AD%98%E7%9A%84)。  

可以用`cudaDevicGetSharedMemConfig(cudaSharedMemConfig *pConfig);`查询存储体大小  
可以用`cudaDeviceSetSharedMemConfig(cudaSharedMemCOnfig config);`设置存储体大小  
参数分别是：
```
cudaSharedMemBankSizeDefault
cudaSharedMemBankSizeFourByte
cudaSharedMemBankSizeEightByte
```

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


> 解决： 文件名应为.cu后缀而不能用.cpp后缀。使用.cu后缀并用nvcc编译时可以不添加cuda.h头。
> 疑问： 如果不添加cuda头文件，如何判定调用的是运行时api还是驱动api

### 无法使用nvprof进行性能分析，因为版本不支持，尝试使用ncu --metrix branch_efficiency进行分析报错
Failed to find metric regex:^branch_efficiency

> 解决： 使用ncu时要求使用基名/全名并且有些指标需要带子后缀（例如.sum/ .avg/.ratio等）可以使用`ncu --query-metrics | grep -i <关键词>`来长沙查询nvprof->ncu的指令对照。
> ncu --metrics支持正则搜索，但也需要先查询基名再使用regex。