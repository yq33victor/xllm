# 自定义算子

## 功能介绍
xllm框架针对以下算子进行了自定义开发和优化：

1. 针对MOE阶段groupMatmul耗时长，优化了GroupMatmul算子的实现。
2. 针对sample阶段中topK和topP在小模型中耗时长的问题，实现topKtopP的融合算子。
3. 针对CPU调度和NPU计算异步并行时，算子琐碎，堵塞异步并行等问题，实现异步token替换算子。

## 用户接口
我们提供两个层面的API：算子调用API和算子直调API。

### 算子调用API
```c++
void top_k_top_p(torch::Tensor& logits,
                 const torch::Tensor& topK,
                 const torch::Tensor& topP);
```

- `logits`: 输入的logits张量，包含模型的输出分数。
- `topK`: 用于选择的前K个概率的阈值张量。
- `topP`: 用于选择的累积概率的阈值张量。

```c++
void replace_token(torch::Tensor& forked, torch::Tensor& lastStepOutPut);
```

- `forked`: 被替换的token张量。
- `lastStepOutPut`: 上一步的输出张量，用于生成新的token。

### 算子直调API
```c++
aclnnStatus aclnnIndexGroupMatmulGetWorkspaceSize(
    const aclTensorList *x,
    const aclTensorList *weight,
    const aclTensorList *scale,
    const aclTensorList *perTokenScale,
    const aclTensor *groupList,
    const aclTensorList *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnIndexGroupMatmul(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

- `x`: 输入的张量列表，包含待处理的数据。
- `weight`: 权重张量，包含模型的参数。
- `scale`: 缩放因子，用于调整输入张量的值。
- `perTokenScale`:每个token的缩放因子，用于动态调整。
- `groupList`: 专家组列表，指示哪些专家参与计算。
- `out`: 输出张量列表，存储计算结果。

## 性能效果

![groupmatmul](../../assets/groupmatmul_performance.png)

* 优化后的GroupMatmul算子在计算时间上表现出明显的优势，尤其是在k为128，m为64情况下，如图所示，优化后算子计算延时 **减少50%**。
* 使用topKtopP融合算子后，在qwen2-0.5B模型中，TTOT **下降37%**,TTFT **提升10%**。
