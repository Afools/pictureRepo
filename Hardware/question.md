### 什么是A+K，什么是A+X。A,K,X指什么 

> 对svm来说，新增需求，粗略分为几个大场景：  
> 接A+K时：支持hccs
> 1. 独立页表，host/device独立页表，性能达到1980的效果
> 2. host/device共页表方案。
> 3. 引入RDMA/SDMA

### PCIe和UB，HCCS的定位有什么区别
- PCIe是传统PCI的演进版本
- NVLink是点对点的GPU间互联技术  
- HCCL与HCCS/CloudMatrix： HCCL（华为集体通信库）是运行于华为昇腾（Ascend）NPU上的软件库，用于实现高效的集体通信 11。HCCL依赖于底层的物理互联，包括用于服务器/节点内芯片间通信的HCCS（华为高速缓存一致性系统/华为计算卡系统）11，以及如CloudMatrix架构中用于多机架NPU集群的大规模光互联架构。

### 什么是IOVA、HPA
IOVA (I/O Virtual Address)  
HPA (Host Physical Address)  
类似如GVA （Guest Virtual Address）

### SOMA Stream Orderd Memory Allocator 的lazy free机制是如何通过内存储接口管理额外占用的内存的
[问题来源](https://jx.huawei.com/community/comgroup/postsDetails?postId=25098779468c43828aa0bbc0786a6a0f&noTop=true&type=freePost&welink_open_uri=aDU6Ly80NzE2NTE3MzE0Nzc5NTcvaHRtbC9pbmRleC5odG1sIy9qeC9kZXRhaWw%2FaWQ9MjUwOTg3Nzk0NjhjNDM4MjhhYTBiYmMwNzg2YTZhMGYmdHlwZT1mcmVlX3Bvc3QmdXJsPQ%3D%3D#:~:text=%E6%89%80%E4%BB%A5%E4%BD%BF%E7%94%A8%E4%B8%80%E5%A5%97%E5%86%85%E5%AD%98%E6%B1%A0%E6%8E%A5%E5%8F%A3%E7%AE%A1%E7%90%86%E8%BF%99%E9%83%A8%E5%88%86%E9%A2%9D%E5%A4%96%E5%8D%A0%E7%94%A8%E7%9A%84%E5%86%85%E5%AD%98%E4%B9%9F%E6%98%AF%E5%8D%81%E5%88%86%E5%90%88%E7%90%86%E5%BE%97%E5%95%A6%E3%80%82)  

### aclGraph起到什么作用，让它capture到MallocAsync有什么作用

### 如何理解aclGraph.Capture解决虚拟地址冲突的方案
[问题来源](https://jx.huawei.com/community/comgroup/postsDetails?postId=25098779468c43828aa0bbc0786a6a0f&noTop=true&type=freePost&welink_open_uri=aDU6Ly80NzE2NTE3MzE0Nzc5NTcvaHRtbC9pbmRleC5odG1sIy9qeC9kZXRhaWw%2FaWQ9MjUwOTg3Nzk0NjhjNDM4MjhhYTBiYmMwNzg2YTZhMGYmdHlwZT1mcmVlX3Bvc3QmdXJsPQ%3D%3D#:~:text=%E6%80%8E%E4%B9%88%E4%BF%9D%E8%AF%81%E5%91%A2%EF%BC%9F-,%E6%9C%80%E7%BB%88%E6%88%91%E4%BB%AC%E7%BB%99%E5%87%BA%E7%9A%84%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88%E6%98%AF,-%EF%BC%8C%E6%88%91%E4%BB%AC%E7%9B%B4%E6%8E%A5)

