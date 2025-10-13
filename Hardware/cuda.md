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
1. **块内线程障碍：**  

`__syncthreads()`  
这是CUDA中最常见的块内同步原语。

- 它是什么？ 一个执行屏障。它要求同一个线程块中的所有线程都必须执行到这个点，才能有任何线程继续执行后面的指令。

- 它的作用： 它同步的是线程的执行流。它确保了一个线程块中的所有线程在代码中的某个特定点“碰头”。

- 它不保证什么？ 仅仅使用 __syncthreads() 并不自动保证一个线程在屏障之前写入的内存（尤其是共享内存）结果，能够被屏障之后的其他线程看到。虽然在实际的CUDA实现中，为了简化编程模型，__syncthreads() 确实包含了一个强大的内存栅栏（如下文所述），但理解其双重角色很重要。

简单比喻： 就像一场团队登山。__syncthreads() 是要求所有队员在半山腰的一个营地集合。在所有队员到达营地之前，任何人都不能继续向上爬。这确保了大家的进度是同步的。

> __syncthreads() 实际上不仅仅是一个执行障碍，它还隐含了一个非常强大的 __threadfence_block() 操作。
>
> 这意味着：
>
> 执行同步：所有线程在此点汇合。  
> 内存同步：它确保线程块内所有线程在 __syncthreads() 之前完成的所有内存操作（写入），对于所有其他线程在 __syncthreads() 之后的操作都是可见的。  
>这就是“对同步前的内存操作可见度不同”的作用所在！

正是因为 __syncthreads() 提供了这个强大的内存栅栏，我们才能安全地用它来协调线程间的数据交换。  

2.  **内存栅栏：**  
```
 __threadfence(),
 __threadfence_block(), 
 __threadfence_system()
```
内存栅栏不同，它同步的是**内存操作（读/写）**的可见性，而不是线程的执行流。

- 它是什么？ 一个内存排序操作。它确保在栅栏指令之前发出的所有内存写入（到全局内存、共享内存甚至本地内存，取决于栅栏的强度）对该栅栏之后发出的操作是可见的。

- 它的作用： 它建立一个“围栏”，防止栅栏后的内存操作（读或写）被GPU或内存系统的硬件优化（如乱序执行、写缓冲）重排到栅栏之前。它不阻止线程继续执行，它只是给内存操作排好队。

类型：

- __threadfence_block(): 确保该线程在栅栏前的内存操作对同一线程块内的其他线程可见。

- __threadfence(): 强度更高。确保对同一GPU上所有线程块的其他线程可见。（常用于全局内存的原子操作前后）

- __threadfence_system(): 强度最高。确保对整个系统（包括CPU和其他GPU） 都可见。（用于多GPU或与CPU协作）

简单比喻： 继续用登山比喻。内存栅栏就像是让一个队员（线程）在营地（同步点）大声喊出他的发现（写入数据），并确保他的喊声已经传播出去并被记录在案（数据可见），然后大家再继续行动。他不需要等其他队员，但他需要确保信息已经传达。 
> 注意！ `__synthreads()`不能滥用，不注意的话很可能造成内核死锁，比如下面这种情况：
```
if (threadID % 2 == 0)
{
  __synthreads();
} else {
  __synthreads();
}
```

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

### 共享内存存储体
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
### 共享内存使用
共享内存可以在核函数声明，也可以在主机端声明，可以动态或静态声明。
1. **核函数内动态声明**  
  核函数内声明如下：
  ```cpp
  __global__ void func(){
    extern __shared__ int list[];
  }
  ```
  这种动态声明需要extern表示这个共享内存是运行时才知道的。调用时通过如下方式：
  ```cpp
  func<<<grid,block,(SIZEOFLIST)*sizeof(int)>>>();
  ```
  第三个参数就是共享内存的大小

### 访问优化方法
**填充**，在声明共享内存时声明额外的填充量，如
```cpp
__shared__ int mem[BDIMY][BDIMX+IPAD];
```
通过添加pad使共享内存中的有效行数据产生交叉，这样在列读取时不会出现多线程访问同一存储体的情况。

## 常量内存
常量内存存储在DRAM上，在SM上有64K的缓存。  
常量内存对内核是**只读**的，主机端可读写。  
常量内存在线程访问同一地址时有广播机制，访问不同地址时进行串行读取，会大大降低访问效率。
### 常量内存使用方法
常量内存用前缀`__constant`在程序头声明，常量内存变量的生命周期与程序周期相同。
常量内存在主机端通过`cudaMemcpyToSymbol()`进行赋值。  

## 线程洗牌指令
线程洗牌是同一线程束的线程直接进行通信的一种方式，可以传递int或float变量。
基本线程洗牌(shuffle)指令函数为
```cpp
int __shfl(int var, int srcLane, int width=warpSize)
```
var 为固定的变量名，srcLane为用width分割线程束得到的段的偏置，如srcLane=2，width=16时，0~15的线程得到2的var值，16~31的线程得到18的值。返回值就是目标线程的var变量的值。

进阶的洗牌指令有`__shfl_up(), __shfl_down(), __shfl_xor()`,具体见[链接](https://face2ai.com/CUDA-F-5-6-%E7%BA%BF%E7%A8%8B%E6%9D%9F%E6%B4%97%E7%89%8C%E6%8C%87%E4%BB%A4/#:~:text=laneID%E9%83%BD%E6%98%AF1-,%E7%BA%BF%E7%A8%8B%E6%9D%9F%E6%B4%97%E7%89%8C%E6%8C%87%E4%BB%A4%E7%9A%84%E4%B8%8D%E5%90%8C%E5%BD%A2%E5%BC%8F,-%E7%BA%BF%E7%A8%8B%E6%9D%9F)  


## CUDA流和并发

流是CUDA支持并发的重要功能，多个流在GPU上并行执行可以有效覆盖GPU数据传输时的空闲时间，隐藏传输时延。  
流声明如下：
```cpp
cudaStream_t streamName;
```
流需要在主机端进行初始化，如下：
```cpp
cudaError_t cudaStreamCreate(cudaStream_t *pStream);   //对流进行初始化
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream=0);  //异步传输数据
```
在流中引入内核函数时需要指定引入的流，如下：
```cpp
cudaKernalFunc<<<grid, block, shareMemSize, streamName>>>(arguments);
```
> <b style="color:pink;">注意，使用异步数据传输时，主机端的内存必须是固定的，而非分页的。</b>  
> 因为主机端和设备端互不知道对方的逻辑内存，在异步执行时会指向固定的物理地址，如果主机端在期间对物理地址进行了重新分配，会导致未定义错误。  
> 分配固定内存（页锁定内存）使用如下方法：
>```cpp
>cudaError_t cudaMallocHost(void** ptr, size_t size);
>cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags);
>```
> 当`cudaHostAlloc`的flags使用`cudaHostAllocDefault`默认值时和`cudaMallocHost()`完全等价。除默认值之外还能取如下值。
> 
>|flags | 含义 | 
>|--|---|
>cudaHostAllocPortable| 分配的内存可被所有 CUDA 上下文访问（默认仅当前上下文可访问）。
>cudaHostAllocMapped | 分配的内存同时在设备端映射（可通过 cudaHostGetDevicePointer 获取设备指针直接访问，无需显式 cudaMemcpy）。
>cudaHostAllocWriteCombined | 分配写合并内存（Write-Combined Memory），牺牲主机端读取性能，提升设备端读取速度（适合主机到设备的单向数据传输）。|
>
> 固定内存的释放要通过`cudaFreeHost()`进行，不能使用`free()`。

内核的最大并发数量根据设备计算能力不同有不同的极限。

## 流调度
在多队列（如Hyper-Q）中可以同时执行多个流，这样减少了单队列时虚假依赖问题（可见[链接](https://face2ai.com/CUDA-F-6-1-%E6%B5%81%E5%92%8C%E4%BA%8B%E4%BB%B6%E6%A6%82%E8%BF%B0/#:~:text=%E5%B9%B6%E5%8F%91%E7%9A%84%E5%85%B3%E9%94%AE-,%E8%99%9A%E5%81%87%E7%9A%84%E4%BE%9D%E8%B5%96%E5%85%B3%E7%B3%BB,-%E5%9C%A8Fermi%E6%9E%B6%E6%9E%84)）。  
在多队列中有些设备支持对流分配优先级，高优先级任务可以抢占低优先级任务资源。优先级流初始化如下:
```cpp
cudaError_t cudaStreamCreateWithPriority(cudaStream *pstream, unsigned int flags, int priority);
```
优先级越高，priority离0越远。

## cuda事件

cuda事件是一种用来对流中执行节点进行标记是工具，能够用来记录事件之间的时间间隔或实现流之间的同步控制。  
事件的声明和初始化、调用方法如下：
```cpp
cudaEvent_t start,stop;                     //声明
cudaCreateEvent(&start);
cudaCreateEvent(&stop);                     //事件初始化

cudaStream_t stream;                        //流声明和初始化
cudaStreamCreate((cudaStream_t*)&stream);

cudaEventRecord(start,stream);              //事件记录和绑定到流，stream参数默认为0
kernel<<<grid,block,0,stream>>>();
cudaEventRecord(stop,stream);

cudaEventSynchronize(stop);                 //等待stop事件触发

float time;
cudaEventElapesdTime(&time,start,stop);     //记录事件间隔

cudaEventDestroy(start);                    //回收事件资源
cudaEventDestroy(stop);

cudaStreamDestroy(stream);
```

## 流同步

流被显示定义时为<b style="color:pink">非空流</b>，没有定义为<b style="color:pink;">空流</b>。空流都是阻塞的，非空流可以通过：
```cpp
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
```
来决定是阻塞的还是非阻塞的，阻塞则为默认`flags=cudaStreamDefault`，非阻塞则`flags=cudaStreamNonBlocking`。
阻塞的流在主机端执行时是非并行的，只有在上一个阻塞流将控制权返回主机端后，下一个阻塞流才会启动。非阻塞流不会被阻塞流阻塞。  

事件提供了一个跨流同步的方式：
```cpp
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudatEvent_t event);
```
指定的流要等待指定的事件，事件完成后流才能继续，事件可以在这个流中，也可以不在。当在不同的流的时候，就实现了跨流同步。  
CDUA提供了一种控制事件行为和性能的函数：
```cpp
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
```
其中参数是：
```
cudaEventDefault
cudaEventBlockingSync
cudaEventDisableTiming
cudaEventInterprocess
```
- `cudaEventBlockingSync`指定使用`cudaEventSynchronize`同步会造成阻塞调用线程。`cudaEventSynchronize`默认是使用cpu周期不断重复查询事件状态  
- `cudaEventBlockingSync`，会将查询放在另一个线程中，而原始线程继续执行，直到事件满足条件，才会通知原始线程，这样可以减少CPU的浪费，但是由于通讯的时间，会造成一定的延迟。  
- `cudaEventDisableTiming`表示事件不用于计时，可以减少系统不必要的开支也能提升cudaStreamWaitEvent和cudaEventQuery的效率  
- `cudaEventInterprocess`表明可能被用于进程之间的事件

### CPU 和 GPU 并行
在非阻塞的流运行中，CPU可以同步进行计算。
```cpp
cudaEvent_t start,stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);
for(int i=0;i<N_SEGMENT;i++)
{
    int ioffset=i*iElem;
    CHECK(cudaMemcpyAsync(&a_d[ioffset],&a_h[ioffset],nByte/N_SEGMENT,cudaMemcpyHostToDevice,stream[i]));
    CHECK(cudaMemcpyAsync(&b_d[ioffset],&b_h[ioffset],nByte/N_SEGMENT,cudaMemcpyHostToDevice,stream[i]));
    sumArraysGPU<<<grid,block,0,stream[i]>>>(&a_d[ioffset],&b_d[ioffset],&res_d[ioffset],iElem);
    CHECK(cudaMemcpyAsync(&res_from_gpu_h[ioffset],&res_d[ioffset],nByte/N_SEGMENT,cudaMemcpyDeviceToHost,stream[i]));
}
//timer
CHECK(cudaEventRecord(stop, 0));
int counter=0;
while (cudaEventQuery(stop)==cudaErrorNotReady)
{
    counter++;
}
printf("cpu counter:%d\n",counter);
```
`cudaEventRecord(stop)`是非阻塞的，cpu可以一直运行。

### 流回调

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