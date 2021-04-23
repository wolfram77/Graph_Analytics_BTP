# Graph_Analytics_BTP

PageRank calculation on given graph aided with cuda programming. <br/>
(*Ongoing project :))

Contributors: <br/>
Dev Lathiya (B18CSE027) <br/>
Mahendra Singh Choudhary (B18CSE028) <br/>

Prof: Dr. Dip Sankar Banerjee

# setup
Install Nvidia cuda toolkit <br/>
create a file (input.txt) <br/>
add testcase into input.txt <br/>

## Output

```bash
Loading graph /root/data/min2c.mtx ...
order: 8 size: 12 {}
kernelCross-3()
computerank()
computerank(): [1] error: 0.0708333 thres: 1e-10
computerank(): [2] error: 0.0602083 thres: 1e-10
computerank(): [3] error: 0.0511771 thres: 1e-10
computerank(): [4] error: 0.0217503 thres: 1e-10
computerank(): [5] error: 0.00616257 thres: 1e-10
computerank(): [6] error: 0.00523819 thres: 1e-10
computerank(): [7] error: 0.00445246 thres: 1e-10
computerank(): [8] error: 0.0018923 thres: 1e-10
computerank(): [9] error: 0.00053615 thres: 1e-10
computerank(): [10] error: 0.000455728 thres: 1e-10
computerank(): [11] error: 0.000387369 thres: 1e-10
computerank(): [12] error: 0.000164632 thres: 1e-10
computerank(): [13] error: 4.66456e-05 thres: 1e-10
computerank(): [14] error: 3.96488e-05 thres: 1e-10
computerank(): [15] error: 3.37015e-05 thres: 1e-10
computerank(): [16] error: 1.43231e-05 thres: 1e-10
computerank(): [17] error: 4.05822e-06 thres: 1e-10
computerank(): [18] error: 3.44949e-06 thres: 1e-10
computerank(): [19] error: 2.93206e-06 thres: 1e-10
computerank(): [20] error: 1.24613e-06 thres: 1e-10
computerank(): [21] error: 3.53069e-07 thres: 1e-10
computerank(): [22] error: 3.00109e-07 thres: 1e-10
computerank(): [23] error: 2.55093e-07 thres: 1e-10
computerank(): [24] error: 1.08414e-07 thres: 1e-10
computerank(): [25] error: 3.07174e-08 thres: 1e-10
computerank(): [26] error: 2.61098e-08 thres: 1e-10
computerank(): [27] error: 2.21933e-08 thres: 1e-10
computerank(): [28] error: 9.43216e-09 thres: 1e-10
computerank(): [29] error: 2.67245e-09 thres: 1e-10
computerank(): [30] error: 2.27158e-09 thres: 1e-10
computerank(): [31] error: 1.93084e-09 thres: 1e-10
computerank(): [32] error: 8.20608e-10 thres: 1e-10
computerank(): [33] error: 2.32506e-10 thres: 1e-10
computerank(): [34] error: 1.9763e-10 thres: 1e-10
computerank(): [35] error: 1.67985e-10 thres: 1e-10
computerank(): [36] error: 7.13937e-11 thres: 1e-10
kernelCross-3()
computerank()
computerank(): [1] error: 0.053125 thres: 1e-10
computerank(): [2] error: 0.0451562 thres: 1e-10
computerank(): [3] error: 0.0383828 thres: 1e-10
computerank(): [4] error: 0.0214421 thres: 1e-10
computerank(): [5] error: 0.0170449 thres: 1e-10
computerank(): [6] error: 0.0144882 thres: 1e-10
computerank(): [7] error: 0.0107399 thres: 1e-10
computerank(): [8] error: 0.00912892 thres: 1e-10
computerank(): [9] error: 0.00775958 thres: 1e-10
computerank(): [10] error: 0.0048359 thres: 1e-10
computerank(): [11] error: 0.00485841 thres: 1e-10
computerank(): [12] error: 0.00412965 thres: 1e-10
computerank(): [13] error: 0.0035102 thres: 1e-10
computerank(): [14] error: 0.00219216 thres: 1e-10
computerank(): [15] error: 0.00186333 thres: 1e-10
computerank(): [16] error: 0.00158383 thres: 1e-10
computerank(): [17] error: 0.00133899 thres: 1e-10
computerank(): [18] error: 0.00114123 thres: 1e-10
computerank(): [19] error: 0.000970046 thres: 1e-10
computerank(): [20] error: 0.000824539 thres: 1e-10
computerank(): [21] error: 0.000591445 thres: 1e-10
computerank(): [22] error: 0.00052106 thres: 1e-10
computerank(): [23] error: 0.000442901 thres: 1e-10
computerank(): [24] error: 0.000376466 thres: 1e-10
computerank(): [25] error: 0.000314367 thres: 1e-10
computerank(): [26] error: 0.000267212 thres: 1e-10
computerank(): [27] error: 0.00022713 thres: 1e-10
computerank(): [28] error: 0.000177327 thres: 1e-10
computerank(): [29] error: 0.000155892 thres: 1e-10
computerank(): [30] error: 0.000132508 thres: 1e-10
computerank(): [31] error: 0.000112632 thres: 1e-10
computerank(): [32] error: 9.41515e-05 thres: 1e-10
computerank(): [33] error: 8.00288e-05 thres: 1e-10
computerank(): [34] error: 6.80244e-05 thres: 1e-10
computerank(): [35] error: 5.58469e-05 thres: 1e-10
computerank(): [36] error: 4.83088e-05 thres: 1e-10
computerank(): [37] error: 4.10625e-05 thres: 1e-10
computerank(): [38] error: 3.49031e-05 thres: 1e-10
computerank(): [39] error: 2.93579e-05 thres: 1e-10
computerank(): [40] error: 2.49542e-05 thres: 1e-10
computerank(): [41] error: 2.12111e-05 thres: 1e-10
computerank(): [42] error: 1.78482e-05 thres: 1e-10
computerank(): [43] error: 1.5248e-05 thres: 1e-10
computerank(): [44] error: 1.29608e-05 thres: 1e-10
computerank(): [45] error: 1.10167e-05 thres: 1e-10
computerank(): [46] error: 9.31153e-06 thres: 1e-10
computerank(): [47] error: 7.9148e-06 thres: 1e-10
computerank(): [48] error: 6.72758e-06 thres: 1e-10
computerank(): [49] error: 5.70521e-06 thres: 1e-10
computerank(): [50] error: 4.85505e-06 thres: 1e-10
computerank(): [51] error: 4.12679e-06 thres: 1e-10
computerank(): [52] error: 3.50777e-06 thres: 1e-10
computerank(): [53] error: 2.97334e-06 thres: 1e-10
computerank(): [54] error: 2.52734e-06 thres: 1e-10
computerank(): [55] error: 2.14824e-06 thres: 1e-10
computerank(): [56] error: 1.82565e-06 thres: 1e-10
computerank(): [57] error: 1.55195e-06 thres: 1e-10
computerank(): [58] error: 1.31916e-06 thres: 1e-10
computerank(): [59] error: 1.12129e-06 thres: 1e-10
computerank(): [60] error: 9.5186e-07 thres: 1e-10
computerank(): [61] error: 8.09266e-07 thres: 1e-10
computerank(): [62] error: 6.87876e-07 thres: 1e-10
computerank(): [63] error: 5.84694e-07 thres: 1e-10
computerank(): [64] error: 4.96934e-07 thres: 1e-10
computerank(): [65] error: 4.22394e-07 thres: 1e-10
computerank(): [66] error: 3.59035e-07 thres: 1e-10
computerank(): [67] error: 3.05003e-07 thres: 1e-10
computerank(): [68] error: 2.59308e-07 thres: 1e-10
computerank(): [69] error: 2.20411e-07 thres: 1e-10
computerank(): [70] error: 1.8735e-07 thres: 1e-10
computerank(): [71] error: 1.5923e-07 thres: 1e-10
computerank(): [72] error: 1.35346e-07 thres: 1e-10
computerank(): [73] error: 1.15044e-07 thres: 1e-10
computerank(): [74] error: 9.77643e-08 thres: 1e-10
computerank(): [75] error: 8.31095e-08 thres: 1e-10
computerank(): [76] error: 7.0643e-08 thres: 1e-10
computerank(): [77] error: 6.00466e-08 thres: 1e-10
computerank(): [78] error: 5.10362e-08 thres: 1e-10
computerank(): [79] error: 4.33808e-08 thres: 1e-10
computerank(): [80] error: 3.68737e-08 thres: 1e-10
computerank(): [81] error: 3.13405e-08 thres: 1e-10
computerank(): [82] error: 2.66403e-08 thres: 1e-10
computerank(): [83] error: 2.26443e-08 thres: 1e-10
computerank(): [84] error: 1.92476e-08 thres: 1e-10
computerank(): [85] error: 1.63599e-08 thres: 1e-10
computerank(): [86] error: 1.39059e-08 thres: 1e-10
computerank(): [87] error: 1.182e-08 thres: 1e-10
computerank(): [88] error: 1.00469e-08 thres: 1e-10
computerank(): [89] error: 8.5399e-09 thres: 1e-10
computerank(): [90] error: 7.25891e-09 thres: 1e-10
computerank(): [91] error: 6.17008e-09 thres: 1e-10
computerank(): [92] error: 5.24447e-09 thres: 1e-10
computerank(): [93] error: 4.4578e-09 thres: 1e-10
computerank(): [94] error: 3.78913e-09 thres: 1e-10
computerank(): [95] error: 3.22076e-09 thres: 1e-10
computerank(): [96] error: 2.73765e-09 thres: 1e-10
computerank(): [97] error: 2.327e-09 thres: 1e-10
computerank(): [98] error: 1.97795e-09 thres: 1e-10
computerank(): [99] error: 1.68124e-09 thres: 1e-10
computerank(): [100] error: 1.42906e-09 thres: 1e-10
computerank(): [101] error: 1.2147e-09 thres: 1e-10
computerank(): [102] error: 1.03249e-09 thres: 1e-10
computerank(): [103] error: 8.7762e-10 thres: 1e-10
computerank(): [104] error: 7.45977e-10 thres: 1e-10
computerank(): [105] error: 6.3408e-10 thres: 1e-10
computerank(): [106] error: 5.38966e-10 thres: 1e-10
computerank(): [107] error: 4.58122e-10 thres: 1e-10
computerank(): [108] error: 3.89404e-10 thres: 1e-10
computerank(): [109] error: 3.30993e-10 thres: 1e-10
computerank(): [110] error: 2.81344e-10 thres: 1e-10
computerank(): [111] error: 2.39142e-10 thres: 1e-10
computerank(): [112] error: 2.03271e-10 thres: 1e-10
computerank(): [113] error: 1.7278e-10 thres: 1e-10
computerank(): [114] error: 1.46863e-10 thres: 1e-10
computerank(): [115] error: 1.24834e-10 thres: 1e-10
computerank(): [116] error: 1.06109e-10 thres: 1e-10
computerank(): [117] error: 9.01923e-11 thres: 1e-10
Ranks:
0.0450409
0.0570347
0.0309304
0.0429898
0.246618
0.123563
0.225233
0.228591

Time taken: 429.186 ms
kernel time: 0.285184 ms

==96== NVPROF is profiling process 96, command: ./a.out /root/data/min2c.mtx 1
==96== Profiling application: ./a.out /root/data/min2c.mtx 1
==96== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   87.18%  173.38us         2  86.687us  84.735us  88.640us  kernelCross(__int64*, __int64*, __int64*, __int64*, __int64*, double*, double*, __int64*, __int64*, __int64*, __int64*, __int64*)
                   11.23%  22.336us        16  1.3960us  1.3440us  1.8240us  [CUDA memcpy HtoD]
                    1.59%  3.1680us         2  1.5840us  1.5680us  1.6000us  [CUDA memcpy DtoH]
      API calls:   99.53%  279.99ms        12  23.332ms  3.7130us  279.94ms  cudaMalloc
                    0.14%  390.72us         1  390.72us  390.72us  390.72us  cuDeviceTotalMem
                    0.08%  229.03us        18  12.723us  4.5760us  44.002us  cudaMemcpy
                    0.06%  180.37us         2  90.187us  88.578us  91.796us  cudaDeviceSynchronize
                    0.06%  174.25us        12  14.521us  3.9350us  95.781us  cudaFree
                    0.06%  166.04us        97  1.7110us     200ns  67.288us  cuDeviceGetAttribute
                    0.03%  90.643us         2  45.321us  32.479us  58.164us  cudaLaunchKernel
                    0.01%  25.614us         1  25.614us  25.614us  25.614us  cuDeviceGetName
                    0.01%  15.688us         4  3.9220us  3.2080us  4.8520us  cudaEventRecord
                    0.00%  13.236us         2  6.6180us  6.4320us  6.8040us  cudaEventSynchronize
                    0.00%  9.0790us         4  2.2690us     683ns  5.8780us  cudaEventCreate
                    0.00%  4.7490us         4  1.1870us     699ns  1.7590us  cudaEventDestroy
                    0.00%  4.1700us         2  2.0850us  1.8290us  2.3410us  cudaEventElapsedTime
                    0.00%  3.6880us         1  3.6880us  3.6880us  3.6880us  cuDeviceGetPCIBusId
                    0.00%  1.5250us         3     508ns     241ns     845ns  cuDeviceGetCount
                    0.00%  1.1450us         2     572ns     293ns     852ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid
```
