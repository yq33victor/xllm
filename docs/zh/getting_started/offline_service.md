# 离线服务

xLLM提供了进行离线推理的主要 Python 接口，即无需使用单独的模型推理服务器即可与模型进行交互。

在`/exmaples`目录下提供了一个简单的脚本，你可以从此开始尝试：
```bash
python examples/generate.py --model='/path/to/Qwen2-7B-Instruct' --devices='npu:0' 
```
该脚本的内容为：
```python title="generate.py"
from xllm import ArgumentParser, LLM, RequestParams

# Create an LLM.
parser = ArgumentParser()
llm = LLM(**vars(parser.parse_args()))

# Create a reqeust params, include sampling params
request_params = RequestParams()
request_params.temperature = 0.8
request_params.top_p = 0.95
request_params.max_tokens = 10
request_params.streaming = False

# Generate texts from the prompts. The output is a list of RequestOutput
# objects that contain the prompt, generated text, and other information.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

outputs = llm.generate(prompts, request_params, None, True)

# Print the outputs.
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```