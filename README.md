# MIT_6.5940
MIT open course,  efficient ML.

You will learn
- model compression, 
- pruning,
- quantization,
- neural architecture search, 
- distributed training, 
- data/model parallelism, 
- gradient compression, 
- on-device fine-tuning. 

## TODO
1. use **auto AWQ**  instead of pseudo-quantization to test AWQ inference speed

## Lab01, 02, 03 Log

The writeup of lab1, 2, 3 is in the corresponding jupyter notebook.


## Lab04 log
I use the hf-mirror(https://hf-mirror.com/) to access the facebook/opt1.5 because of the gfw in China.
And I transfer jupyter to python normal file.

The log is in final_lab4_log.txt. We can see AWQ alleviate the error caused by 3bit quantization, from 0 to 8. 

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


## Lab05 log
My code is in  `/MIT_6.5940/tinychat-tutorial/kernels/starter_code`

### Evaluation

| Section                     | Total time(ms) | Average time(ms) | Count | GOPs (giga operations per seconds)     |
|------------------------------|-----------------|-------------------|-------|----------|
| reference                   | 2510.913086     | 251.091003        | 10    | 1.044019 |
| loop_unrolling              | 1979.465942     | 197.945999        | 10    | 1.324317 |
| multithreading              | 799.554993      | 79.955002         | 10    | 3.278624 |
| simd_programming            | 1581.666016     | 158.166000        | 10    | 1.657392 |
| multithreading_loop_unrolling| 586.346008      | 58.633999         | 10    | 4.470807 |
| all_techniques              | 186.626999      | 18.662001         | 10    | 14.046413 |

### Report


1. How does the performance in GOPs, achieved through loop unrolling on your computer, compare to the reference implementation? Please explain the performance difference. (5pt)
-  1.044 vs 1.324, loop unrolling is faster because of the CPU pipeline and reduction of for (branch).




3. How does the performance in GOPs, achieved through multithreading on your computer, compare to the reference implementation? Please explain the performance difference. (5pt)
- 1.04 vs 3.27, the thread we used is four, which accelerate the computation by 3.x times.



4. How does the performance in GOPs, achieved through SIMD programming on your computer, compare to the reference implementation? Please explain the performance difference. (5pt)
- 1.04 vs 1.657392, SIMD compact four integer into one computation. Thus, the ratio is not as high as multi-thread. 



5. How does the performance in GOPs, achieved through multithreading and loop unrolling on your computer, compare to the reference implementation? Please explain the performance difference. (5pt)
- 1.04 vs 4.47. This ratio is the multiple of multithreading and loop unrolling.



6. How does the performance in GOPs, achieved through all optimization techniques on your computer, compare to the reference implementation? Please explain the performance difference. (5pt)
- 1.04 vs 14.04. This is really impressive.



7.Bonus (20pt): Any optimization techniques on your mind? Try to implement them to improve the performance further! If you can further improve the performance compared to the optimized kernel in TinyChatEngine, you can get bonus points here! Each percent of performance speedup equals one point (create a pull request in the repo and get verified by the TA), up to 20 points.

### terminal LOG
```bash
$ ./evaluate.sh reference
-------- Sanity check of reference implementation: Passed! -------- 
Section, Total time(ms), Average time(ms), Count, GOPs
reference, 2510.913086, 251.091003, 10, 1.044019

All tests completed!
$ ./evaluate.sh loop_unrolling
-------- Sanity check of loop_unrolling implementation: Passed! -------- 
Section, Total time(ms), Average time(ms), Count, GOPs
loop_unrolling, 1979.465942, 197.945999, 10, 1.324317

All tests completed!
$ ./evaluate.sh multithreading
-------- Sanity check of multithreading implementation: Passed! -------- 
Section, Total time(ms), Average time(ms), Count, GOPs
multithreading, 799.554993, 79.955002, 10, 3.278624

All tests completed!
$ ./evaluate.sh simd_programming
-------- Sanity check of simd_programming implementation: Passed! -------- 
Section, Total time(ms), Average time(ms), Count, GOPs
simd_programming, 1581.666016, 158.166000, 10, 1.657392

All tests completed!
$ ./evaluate.sh multithreading_loop_unrolling
-------- Sanity check of multithreading_loop_unrolling implementation: Passed! -------- 
Section, Total time(ms), Average time(ms), Count, GOPs
multithreading_loop_unrolling, 586.346008, 58.633999, 10, 4.470807

All tests completed!
$ ./evaluate.sh all_techniques
-------- Sanity check of all_techniques implementation: Passed! -------- 
Section, Total time(ms), Average time(ms), Count, GOPs
all_techniques, 186.626999, 18.662001, 10, 14.046413

All tests completed!
```
