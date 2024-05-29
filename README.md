# MIT_6.5940
MIT open course,  efficient ML

## Lab04 log

Due to the gfw in China, I use the hf-mirror(https://hf-mirror.com/) to access the facebook/opt1.5, and transfer the jupyter to python file.
The log is in final_lab4_log.txt. We can see AWQ did relief the error caused by 3bit quantization, from 0 to 8. 

Note: the quantization in Lab4 is pseudo quantization. The quantization of weight will be dequantized during inference. The error happens in the quantization and dequantization of weights.

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
