# 全异步多流运行时


## 功能介绍

xLLM引擎层结合异构硬件的特性，设计了全异步运行时，具体包括3层流水线：
- 框架层-异步调度
    - 通过解耦CPU调度与device计算，在 step-i 计算的同时同步执行 step-i+1 的调度操作，最小化两个step之间device等待的空泡。 
- 模型图层-双流并行
    - 通过拆分两个micro batches并分配不同的流，一个流执行计算，另一个流执行通信，实现计算与通信的高效重叠，有效降低延时。 
- 算子层-kernel并行
    - 通过采用多流，将向量运算kernel与矩阵乘kernel分配至不同流上，同一时刻分别使用不同的计算单元执行，充分利用全部硬件单元。


## 使用方式

上述功能已经在xllm引擎内部进行了实现，均对用户透明，用户无需关注内部实现细节，在适用的场景直接开启相关功能即可。
针对异步调度和双流并行，我们分别提供了两个gflags参数，`enable_schedule_overlap`、`enable_comp_comm_overlap`，这两个参数默认均为false，如需开启在xllm的服务启动脚本中设置为true即可，示例如下：
```shell
--enable_schedule_overlap=true
--enable_comp_comm_overlap=true
```


## 性能效果
- 异步调度开启后，两个step之间的调度开销在200us左右，基本类似一个kernel launch的时间，在DeepSeek-R1-Distill-Qwen-1.5B模型上，限制TPOT 50ms，吞吐**提升17%**。
- prefill双流并行开启后，基本可掩盖75以上的通信开销，在DeepSeek-R1模型上，只输出1个token的情况下，TTFT下降7%，吞吐**提升7%**。


## 注意事项
- 异步调度功能会在服务端额外计算一个step，当使用场景中输出token数量较少，或是类似embedding模型只一次性输出的场景，不建议开启，会影响服务端吞吐。
- 双流并行目前只支持prefill阶段，请求输入越长，收益越大。