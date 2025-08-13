# 多种调度器

##功能介绍
xLLM实现了多种请求调度策略，支持continuous batching、chunked prefill、基于模拟的zero evict等batch策略，同时全面支持PD分离场景。
同时对于非PD分离场景，都支持prefix_cache匹配。prefix_cache基于mermer_hash，使用lru淘汰策略，提供更极致的匹配效率，同时提高prefix_cache命中率。

## 使用方式
上述策略已在xLLM实现，并向外暴露`gflag`参数，控制功能的开关。

- 开启基于mermer_hash实现的prefix_cache。
```
--prefix_cache_policy=murmur_hash3
```

- 开启prefix_cache。
```
--enable_prefix_cache=true
```

- 开启chunked prefill，并设置chunked_size。
```
--enable_chunked_prefill=true
--max_tokens_per_chunk_for_prefill=256
```

## 性能效果
开启prefix_cache之后，在`Qwen3-8B`模型上，限制E2E 10s，吞吐可 **提升16%**。
开启chunked_prefill之后，在`Qwen3-8B`模型上，限制TPOT 50ms，TTFT **下降46%**。

## 注意事项
PD分离暂不支持prefix_cache，chunked_prefill功能，使用时需关闭，设置为`false`。