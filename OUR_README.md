# Diffusion GRPO

```
ACCELERATE_LOG_LEVEL=info     
accelerate launch 
    --config_file recipes/accelerate_configs/zero3_singlegpu.yaml     
    src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distil l-Qwen-1.5B/grpo/config_demo.yaml
    --vllm_mode colocate
```