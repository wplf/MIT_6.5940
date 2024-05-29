# Lab04 log
|strategy|perplexity|
|--|--|
|float|14.47|
|naive 4bit quantization of weight|121.90|
|protect salient weight (1%)|  17.15|
|random protect 1% |121.654|
|AWQ scale = 1 |121.90|
|AWQ scale = 2 |18.93|
|AWQ scale = 3 |19.25|
|AWQ scale = 4 |21.26|

further job need to do:
1. use **auto AWQ**  instead of pseudo-quantization to test AWQ inference speed


```text

evaluating...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:41<00:00,  1.03s/it]
####################evaluating float perplexity and model size
model perplexity: 14.47
model size: 5043.73 MiB

evaluating...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:39<00:00,  1.01it/s]
####################evaluating 4 bit model perplexity and model size (weight only), and use pseudo quantization (quantize and dequantize)
model perplexity: 121.90
model size: 495.06 MiB

Collecting activation scales...
Downloading readme: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 167/167 [00:00<00:00, 1.61MB/s]
Repo card metadata block was not found. Setting CardData to empty.
 * Split into 127 blocks
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 127/127 [00:24<00:00,  5.13it/s]


evaluating...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:38<00:00,  1.03it/s]
Processing 1% salient weight channels, and model perplexity is 17.152280807495117


evaluating...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:38<00:00,  1.04it/s]
Processing random weight quantization, and model perplexity is 121.65473937988281

# AWQ
#################### processing scale factor:  1
evaluating...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:38<00:00,  1.04it/s]
 Processing AWQ weight quantization, and  model perplexity: 121.90
model size: 495.06 MiB
#################### processing scale factor:  2
evaluating...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:38<00:00,  1.04it/s]
Processing AWQ weight quantization, and  model perplexity: 18.93
model size: 495.06 MiB
#################### processing scale factor:  3
evaluating...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:39<00:00,  1.01it/s]
Processing AWQ weight quantization, and  model perplexity: 19.25
model size: 495.06 MiB
#################### processing scale factor:  4
evaluating...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:40<00:00,  1.01s/it]
Processing AWQ weight quantization, and  model perplexity: 21.26

```