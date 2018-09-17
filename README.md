# Assignment 2: Image blurring

Assignment No 2 for the multi-core programming course. Using previous code examples, blur an image using OpenCV and CUDA. It has to be programmed in three ways:

- In CPU without threads.
- In CPU with threads.
- In CUDA using blocks and threads.

For the CPU version with threads, test performance varying the number of threads depending on your CPU. For the GPU version, test the performance with different thread configurations. Use the grid and blocks configuration that achieved the best performance from the matrix multiplication assignment.

Include a Pdf file with the results of the testing for each case. Measure the time spent for the calculations, and the overall time of each code. Include the characteristics of the computer where the testing was performed; mention the type, speed, number of cores, etc, both for the CPU and the GPU. Add conclusions and thoughts after analyzing the results.

Rubric:

1. The program shows the original image, and the blurred image. *Complete*
2. Applied a 5x5 bluring window. *Incomplete*
3. Images are loaded and displayed correctly. *Complete*
4. GPU code is initialized correctly. *Complete*
5. The report file has tables with the performance data for the different configurations, as well as for the speedup obtained. *Incomplete*
6. The report file has the computer's characteristics, as well as the conclusions. *Complete*

**NOTES**

1. There was an error with the blocks and threads. If you set *int xBlock = 16* and *int yBlock = 1024*, you exceed the number of threads per block, and the kernel does not execute properly. Changed it to *int xBlock = 16* and *int yBlock = 64* to avoid the issue.
2. Speedups are missing.

**Grade: 90**