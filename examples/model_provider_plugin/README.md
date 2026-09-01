# External model provider PoC

Install the provider beside FreeToken, then explicitly allow its distribution:

```bash
pip install -e examples/model_provider_plugin
FREETOKEN_MODEL_PROVIDERS=freetoken-example-model-provider ft serve MODEL_PATH
```

The model config must name `ExampleLlamaForCausalLM` in `architectures`. This
example reuses FreeToken's Llama implementation; a real provider can point the
same `ModelSpec` at its own model module.
