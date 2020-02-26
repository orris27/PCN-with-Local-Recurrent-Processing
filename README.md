# FLOPs

### FLOPs

#### CIFAR10

In our model, "127,796,480 " in "127,796,480 + 2,560" means the flops with no feedbacks, while "2,560" means if we add 1 circles in our model, 2560 flops will be added to our model.

| Classifiers                       | FLOPs                       | params                  |
| --------------------------------- | --------------------------- | ----------------------- |
| PCN                               |                             |                         |
| circles=0                         | 547,230,720                 | 5,212,816               |
| circles=1                         | 1,039,733,760               | 9,898,192               |
| circles=2                         | 1,532,236,800               | 9,898,192               |
| Ours                              |                             |                         |
| Simple                            |                             |                         |
| exit-1                            | 127,796,480 + 2,560         | 128,026                 |
| exit-2                            | 253,628,260 + 7,880         | 626,176                 |
| exit-3                            | 547,234,760 + 18,320        | 5,227,750               |
| Linear Hidden (modelC_h_dp2)      |                             |                         |
| exit-1                            | 127799616 + 8192            | 129840 (134096)         |
| exit-2                            | 253647808 + 45056           | 641658 (664698)         |
| exit-3                            | 547,286,592 + 118,784       | 5,269,956 (5,330,500)   |
| modelG\|0:hidden;1:fb+hidden;2:no |                             |                         |
| exit-1                            | 127799616 + 0               | 129796                  |
| exit-2                            | 253648608 + 1440            | 642314 (643116)         |
| exit-3                            | 547255008 + 1440            | 5238056 (5238858)       |
| Conv Hidden (128, 256, 512)       |                             |                         |
| exit-1                            | 295,568,640 + 150,994,944   | 290,448 (438,032)       |
| exit-2                            | 673,058,560 + 377,487,360   | 1,768,602 (2,801,178)   |
| exit-3                            | 1,218,323,200 + 603,979,776 | 10,296,484 (14,868,516) |
| Conv Hidden (32, 64, 64)          |                             |                         |
| exit-1                            | 169,738,560 + 37,748,736    | 166,416 (203,312)       |
| exit-2                            | 342,754,240 + 80,216,064    | 843,546 (1,046,394)     |
| exit-3                            | 659,949,120 + 101,449,728   | 5,802,532 (6,337,220)   |

#### CIFAR100



| Classifiers                       | FLOPs                         | params                  |
| --------------------------------- | ----------------------------- | ----------------------- |
| PCN                               |                               |                         |
| circles=0                         | 547,276,800                   | 5,258,986               |
| circles=1                         | 1,039,779,840                 | 9,944,362               |
| circles=2                         | 1,532,282,880                 | 9,944,362               |
| PCN: Linear Hidden (prednet_h)    |                               |                         |
| T=0                               | 547,264,768                   | 5,247,146               |
| Ours                              |                               |                         |
| Simple                            |                               |                         |
| exit-1                            | 127,808,000 + 2,560           | 138,218 (151,246)       |
| exit-2                            | 253,672,720 + 7,880           | 666,462 (715,546)       |
| exit-3                            | 547,335,200 + 18,320          | 5,318,354 (5,429,350)   |
| Linear Hidden (modelC_h_dp2)      |                               |                         |
| exit-1                            | 127,802,496 + 8,192           | 132,810 (137,066)       |
| exit-2                            | 253,656,448 + 45,056          | 650,478 (673,518)       |
| exit-3                            | 547,300,992 + 118,784         | 5,284,626 (5,345,170)   |
| modelG\|0:hidden;1:fb+hidden;2:no |                               |                         |
| exit-1                            | 127802496 + 0                 | 132946                  |
| exit-2                            | 253664448 + 1440              | 658694 (666066)         |
| exit-3                            | 547316928 + 1440              | 5300786 (5308158)       |
| Conv Hidden (128, 256, 512)       |                               |                         |
| exit-1                            | 295,580,160 + 150,994,944     | 302,058 (449642)        |
| exit-2                            | 673,093,120 + 377,487,360     | 1,803,342 (2,835,918)   |
| exit-3                            | 1,218,403,840 + 603,979,776   | 10,377,394 (14,949,426) |
| Conv Hidden (32, 64, 64)          |                               |                         |
| exit-1                            | 169,741,440 + 37,748,736 * T  | 169,386 (206,282)       |
| exit-2                            | 342,762,880 + 80,216,064 * T  | 852,366 (1,055,214)     |
| exit-3                            | 659,963,520 + 101,449,728 * T | 5,817,202 (6,351,890)   |





### CIFAR10

For PCN, "94.39 (94.77)" means the reported accuracy in paper is 94.39, while 94.77 is the accuracy obtained by running their codes.

| Setting                            | Acc1  | Acc2  | Acc3          |
| ---------------------------------- | ----- | ----- | ------------- |
| PCN                                |       |       |               |
| T=0                                |       |       | 94.39 (94.77) |
| T=1                                |       |       | 94.27 (94.35) |
| T=2                                |       |       | 94.61         |
| Ours                               |       |       |               |
| vanilla                            | 80.85 | 90.58 | 93.37         |
| vanilla (all35clf15)               | 80.87 | 90.85 | 93.47         |
| T=0                                | 80.96 | 90.19 | 93.09         |
| T=1                                | 83.47 | 91.05 | 92.82         |
| T=1 (val no fb)                    | 77.13 | 89.74 | 92.08         |
| T=1 (val no fb, dp0.5)             | 80.82 | 91.12 | 93.87         |
| T=1 (val no fb, dp0.5, all15clf10) | 80.24 | 91.11 | 93.98         |
| T=1 (val no fb, dp0.5, all35clf15) | 81.10 | 91.24 | 94.24         |
| T=1 (val no fb, dp0.3)             | 81.30 | 91.02 | 93.49         |
| T=2 (val no fb, dp0.5)             | 80.96 | 90.89 | 93.49         |







### CIFAR100

#### VGG 

##### Linear Classifiers

| Setting                                | Acc1  | Acc2  | Acc3          | Adaptive Acc                    |
| -------------------------------------- | ----- | ----- | ------------- | ------------------------------- |
| PCN                                    |       |       |               |                                 |
| T=0                                    |       |       | 74.69 (75.11) |                                 |
| T=1                                    |       |       | 76.22 (76.34) |                                 |
| T=2                                    |       |       | 77.25 (77.44) |                                 |
| Ours                                   |       |       |               |                                 |
| vanilla                                | 49.57 | 68.33 | 73.01         | 69.11(0.353 0.401 0.246, ts0.5) |
| vanilla (all35clf15)                   | 48.55 | 68.11 | 72.74         | 68.96(0.339 0.403 0.258, ts0.5) |
| T=0                                    | 50.24 | 67.82 | 72.99         |                                 |
| T=0 (ge)                               | 49.80 | 67.45 | 73.28         |                                 |
| T=1                                    | 56.91 | 68.30 | 72.14         |                                 |
| T=1 (ge)                               | 55.33 | 67.82 | 72.49         |                                 |
| T=1 (val no fb)                        | 24.07 | 54.96 | 67.48         |                                 |
| T=1 (val no fb, ge)                    | 23.56 | 52.40 | 68.56         |                                 |
| T=1 (val no fb, dp0.5)                 | 47.17 | 67.26 | 72.37         |                                 |
| T=1 (val no fb, dp0.5, ge, 001)        | 48.68 | 67.68 | 73.27         | 68.82(0.343 0.412 0.245, ts0.5) |
| T=1 (val no fb, dp0.5, ge, 010)        | 49.24 | 66.68 | 72.96         | 68.81(0.356 0.327 0.316, ts0.5) |
| T=1 (val no fb, dp0.5, ge, 011)        | 49.00 | 66.61 | 72.80         | 69.20(0.347 0.341 0.312, ts0.5) |
| T=1 (val no fb, dp0.5, ge, 100)        | 48.30 | 67.16 | 73.24         | 70.16(0.256 0.477 0.267, ts0.5) |
| T=1 (val no fb, dp0.5, ge, 101)        | 47.39 | 67.55 | 73.68         | 70.25(0.249 0.473 0.277, ts0.5) |
| T=1 (val no fb, dp0.5, ge, 110)        | 46.33 | 66.87 | 73.00         | 70.38(0.241 0.395 0.364, ts0.5) |
| T=1 (val no fb, dp0.5, ge, 111)        | 47.49 | 67.09 | 73.48         |                                 |
| T=1 (val no fb, dp0.5, all35clf15)     | 47.25 | 67.25 | 72.87         |                                 |
| T=1 (val no fb, dp0.5, all35clf15, ge) | 46.12 | 66.77 | 73.17         |                                 |
| SettingT=1 (val no fb, dp0.3)          | -     | -     | -             |                                 |
| T=2 (val no fb, dp0.5)                 | -     | -     | -             |                                 |

##### Two-layer Linear Classifiers (modelC_h_dp2)

| Setting | Acc1   | Acc2   | Acc3   | Adaptive Acc (ts0.5)        |
| ------- | ------ | ------ | ------ | --------------------------- |
| PCN     |        |        |        |                             |
| T=0     |        |        | 75.560 |                             |
| Ours    |        |        |        |                             |
| T=0     | 52.940 | 66.830 | 73.850 | 64.720 \| 0.569 0.300 0.131 |
| T=2     | 54.690 | 67.190 | 74.720 | 66.570 \| 0.568 0.300 0.132 |
| T=3     | 53.750 | 68.390 | 74.730 | 66.230 \| 0.562 0.312 0.126 |
| T=4     | 53.350 | 68.080 | 74.910 | 66.500 \| 0.556 0.324 0.120 |
| T=5     | 53.760 | 67.870 | 75.330 | 66.150 \| 0.562 0.327 0.111 |
| T=6     | 53.530 | 68.800 | 75.230 | 66.850 \| 0.567 0.331 0.102 |
| T=8     | 53.200 | 68.220 | 74.510 | 65.510 \| 0.563 0.337 0.100 |
| T=10    | 53.210 | 69.060 | 74.950 | 66.300 \| 0.552 0.366 0.083 |

###### variance

| Setting | Acc1    | Acc2   | Acc3   |
| ------- | ------- | ------ | ------ |
| T=5     |         |        |        |
|         | 53.760  | 67.870 | 75.330 |
|         | 53.030, | 68.380 | 75.070 |
|         | 53.040  | 68.180 | 74.590 |
|         | 53.070  | 68.660 | 75.020 |
|         | 54.420  | 68.870 | 74.820 |
| T=2     |         |        |        |
|         | 54.690  | 67.190 | 74.720 |
|         | 54.280  | 66.850 | 75.050 |
|         | 54.350  | 66.490 | 75.170 |





###### Different Thresholds

T=5 （75.33）

| threshold | exit-1 | exit-2 | exit-3 | Adaptive Acc | FLOPs       |
| --------- | ------ | ------ | ------ | ------------ | ----------- |
| 0.5       | 0.562  | 0.327  | 0.111  | 66.150       |             |
| 0.6       | 0.451  | 0.365  | 0.184  | 69.330       |             |
| 0.7       | 0.362  | 0.378  | 0.260  | 71.640       |             |
| 0.8       | 0.277  | 0.382  | 0.341  | 73.160       | 319,185,617 |
| 0.85      | 0.230  | 0.378  | 0.392  | 73.940       | 340,183,905 |
| 0.9       | 0.180  | 0.368  | 0.452  | 74.600       | 364,013,198 |
| 0.92      | 0.159  | 0.362  | 0.479  | 74.760       | 374,741,170 |
| 0.95      | 0.121  | 0.347  | 0.532  | 74.940       | 394,995,695 |
| 0.98      | 0.064  | 0.309  | 0.627  | 75.270       | 430,211,960 |



##### No Hidden Layer (modelC_dp)

| Setting | Acc1 | Acc2 | Acc3 | Adaptive Acc (ts0.5) |
| ---- | ---- | ---- | ---- | ---- |
| PCN |  |  |  |  |
| T=0 |  |  | 75.63 |  |
| Ours |  |  |  |  |
| T=0      | 56.600 | 66.950 | 72.340 | 67.720 \| 0.540 0.282 0.178 |
| T=2      | 56.420 | 68.700 | 74.470 | 67.680 \| 0.588 0.257 0.155 |
| T=3      | 56.460 | 67.940 | 74.110 | 67.100 \| 0.600 0.272 0.128 |
| T=4      | 56.920 | 68.980 | 75.400 | 67.710 \| 0.598 0.282 0.120 |
| T=5      | 55.640 | 68.360 | 75.050 | 66.870 \| 0.600 0.291 0.109 |
| T=6      | 56.050 | 69.430 | 75.080 | 67.030 \| 0.605 0.297 0.098 |
| T=8      | 55.950 | 69.560 | 75.160 | 66.620 \| 0.604 0.315 0.082 |
| T=10      | 56.100 | 69.000 | 74.750 | 66.740 \| 0.576 0.339 0.085 |



###### random for modelC_dp2

+ Acc1 = 55.640

+ Acc2 = 68.360
+ Acc3 = 75.050

| Setting   | exit-1 | exit-2 | exit-3 | Adaptive Acc |
| --------- | ------ | ------ | ------ | ------------ |
| random    |        |        |        |              |
|           | 0.331  | 0.338  | 0.331  | 66.390       |
| 0.5       | 0.605  | 0.287  | 0.109  | 61.200       |
| 0.6       | 0.498  | 0.331  | 0.171  | 63.720       |
| 0.7       | 0.406  | 0.360  | 0.234  | 64.570       |
| 0.8       | 0.330  | 0.353  | 0.317  | 66.280       |
| 0.9       | 0.235  | 0.357  | 0.408  | 67.850       |
| 0.95      | 0.164  | 0.340  | 0.496  | 69.590       |
| threshold |        |        |        |              |
| 0.5       | 0.600  | 0.291  | 0.109  | 66.870       |
| 0.6       | 0.500  | 0.329  | 0.171  | 69.660       |
| 0.7       | 0.411  | 0.354  | 0.235  | 71.700       |
| 0.8       | 0.332  | 0.359  | 0.308  | 73.250       |
| 0.9       | 0.234  | 0.360  | 0.406  | 74.180       |
| 0.95      | 0.167  | 0.341  | 0.492  | 74.670       |



##### Convolution Classifier (32-64-64)

| Setting                                                      | Acc1  | Acc2  | Acc3  | Adaptive Acc                     |
| ------------------------------------------------------------ | ----- | ----- | ----- | -------------------------------- |
| PCN                                                          |       |       |       |                                  |
| T=0                                                          |       |       | 76.23 |                                  |
| Ours                                                         |       |       |       |                                  |
| T=0 (val no fb, dp0.5, ge)                                   | 49.46 | 67.46 | 72.93 | 66.46(0.445 0.412 0.142, ts0.5)  |
| T=0 (ge, bn)                                                 | 54.11 | 69.44 | 74.60 | 69.48 (0.474 0.344 0.182, ts0.5) |
| T=1(ge, bn)                                                  | 57.98 | 70.31 | 75.81 | 69.77 (0.563 0.289 0.148, ts0.5) |
| T=1 (val no fb, dp0.5, ge)                                   | 46.22 | 66.56 | 73.09 | 67.68(0.342 0.447 0.211, ts0.5)  |
| T=1(val no fb, dp0.5, ge, bn)                                | 49.94 | 60.27 | 74.26 | 72.54 (0.252 0.147 0.602, ts0.5) |
| T=1(val no fb, dp0.5, ge, bn, main-fb)                       | 47.25 | 60.59 | 71.02 | 68.63 (0.270 0.220 0.510, ts0.5) |
| T=1 (val no fb, dp0.5, ge, bn, lmbda0.01)\| T\exit \| 0      \| 1      \| 2      \| Adaptive (ts0.5)            \| | 51.96 | 69.17 | 74.45 | 71.17 (0.339 0.429 0.232, ts0.5) |
| T=1 (ge, bn, lmbda0.01)                                      | 52.78 | 69.58 | 74.49 | 69.36 (0.468 0.359 0.173,ts0.5)  |
| T=1 (val no fb, dp0.5, bn, lmbda0.01)                        | 53.56 | 69.59 | 73.96 | 70.17 (0.401 0.401 0.198, ts0.5) |
| T=2 (ge, bn)                                                 | 58.63 | 69.92 | 75.34 | 68.99 (0.595 0.283 0.123, ts0.5) |
| T=2 (val no fb, dp0.5, ge)                                   | 46.89 | 66.44 | 73.61 | 67.31(0.370 0.432 0.198, ts0.5)  |
| T=2 (dp2, ge)                                                | -     | -     | -     | -                                |
| T=3 (dp2, ge)                                                | -     | -     | -     | -                                |
| T=4 (dp2, ge)                                                | -     | -     | -     | -                                |
| T=5 (ge, bn)                                                 | 59.98 | 71.09 | 76.13 | 69.05 (0.647 0.271 0.082, ts0.5) |
| T=5(val no fb, dp0.5, ge, bn)                                | 26.23 | 21.48 | 72.13 | 73.26(0.055 0.003 0.942, ts0.5)  |
| T=5 (val no fb, dp0.5, ge, bn, lmbda0.01)                    | 54.00 | 69.46 | 74.50 | 69.33(0.459 0.356 0.185, ts0.5)  |
| T=5 (val no fb, dp0.5, ge, bn, lmbda0.1)                     | 54.65 | 69.05 | 73.82 | 69.16(0.466 0.357 0.177, ts0.5)  |
| T=5 (dp2, ge)                                                | -     | -     | -     | -                                |
| T=10(val no fb, dp0.5, ge, bn, lmbda0.01)                    | 54.85 | 69.13 | 74.15 | 69.17(0.474 0.349 0.177, ts0.5)  |
| T=10 (dp2, ge)                                               | -     | -     | -     | -                                |

##### Convolution Classifier (32-64-64) + Feedback in Backbone (modelF)

| Setting                    | Acc1  | Acc2  | Acc3  | Adaptive Acc                     |
| -------------------------- | ----- | ----- | ----- | -------------------------------- |
| PCN                        |       |       |       |                                  |
| T=0                        |       |       | 76.57 |                                  |
| T=1                        |       |       | 77.57 |                                  |
| T=2                        |       |       | 78.25 |                                  |
| Ours                       |       |       |       |                                  |
| T=1 (val no fb, dp0.5, ge) | 47.25 | 60.59 | 71.02 | 68.63 (0.270 0.220 0.510, ts0.5) |
| T=1 (ge)                   | 62.75 | 72.32 | 76.64 | 71.03 (0.653 0.233 0.114, ts0.5) |
| T=2 (ge)                   | 63.85 | 73.03 | 77.88 | 71.51 (0.691 0.209 0.100, ts0.5) |
| T=5 (ge)                   | 63.83 | 73.66 | 77.88 | 70.60 (0.754 0.183 0.062, ts0.5) |



##### modelG

| Setting               | Acc1    | Acc2   | Acc3   | Adaptive Acc                |
| --------------------- | ------- | ------ | ------ | --------------------------- |
| baseline              |         |        | 76.070 |                             |
| ce                    |         |        |        |                             |
| T=0                   | 53.140  | 67.780 | 73.600 | 65.510 \| 0.572 0.309 0.119 |
| T=0 (2con3)           | 53.280  | 67.520 | 73.700 | 65.510 \| 0.574 0.299 0.127 |
| T=0 (2con3, se)       | 55.740  | 66.210 | 73.800 | 66.570 \| 0.571 0.289 0.139 |
| T=2 (2con3)           | 53.530  | 66.990 | 74.070 | 65.560 \| 0.578 0.297 0.125 |
| T=2 (2con3, se)       | 54.540, | 67.570 | 74.800 | 66.570 \| 0.579 0.292 0.129 |
| T=3                   | 54.000  | 66.030 | 74.020 | 65.850 \| 0.565 0.302 0.133 |
| T=5                   | 53.620  | 67.400 | 73.970 | 66.270 \| 0.572 0.298 0.130 |
| T=5 (0con1, 2con3)    | 53.860  | 68.480 | 74.450 | 66.230 \| 0.573 0.324 0.104 |
| T=5 (2con3)           | 53.270  | 69.070 | 74.800 | 66.490 \| 0.573 0.329 0.098 |
| T=5 (2con3, se)       | 53.700  | 67.380 | 74.530 | 66.110 \| 0.566 0.330 0.104 |
| T=8                   | 53.620  | 67.420 | 74.100 | 65.400 \| 0.567 0.311 0.122 |
| T=8 (2con3)           | 53.390  | 68.220 | 75.320 | 65.960 \| 0.580 0.330 0.090 |
| T=8 (2con3, se)       | 54.430  | 67.200 | 75.030 | 65.440 \| 0.581 0.320 0.099 |
| T=10                  | 53.220  | 67.650 | 73.910 | 64.950 \| 0.577 0.297 0.126 |
| ee (lmbda1)           |         |        |        |                             |
| T=0                   | 1.000   | 5.380  | 76.160 | 12.330 \| 0.000 0.882 0.118 |
| T=3                   | 1.000   | 8.580  | 76.650 | 15.690 \| 0.000 0.880 0.120 |
| T=5                   | 1.000   | 8.320  | 76.530 | 15.320 \| 0.000 0.882 0.118 |
| T=5 (2 cross entropy) | 51.440  | 67.680 | 72.130 | 67.570 \| 0.007 0.993 0.000 |
| T=8                   | 1.000   | 6.810  | 76.390 | 15.780 \| 0.000 0.852 0.148 |
| T=10                  | 1.920   | 6.080  | 76.560 | 13.550 \| 0.000 0.872 0.128 |















#### ResNet56

Default Setting: Gradient Equilibrium + No knowledge distillation

kd setting:

1. T3.0,gamma0.9

##### No Hidden Layer

| Setting  | Acc1    | Acc2    | Acc3                                                         | Adaptive Acc                           |
| -------- | ------- | ------- | ------------------------------------------------------------ | -------------------------------------- |
| Baseline |         |         |                                                              |                                        |
| resnet |         |         | 71.020                                                       |                                        |
| Ours     |         |         |                                                              |                                        |
| T=0    | 44.070 | 64.570 | 67.290 | 65.280 \| 0.270 0.440 0.291 |
| T=0 (kd1) | 24.890 | 43.150 | 71.720 | 70.950 \| 0.016 0.100 0.884 |
| T=2    | 44.660 | 65.320 | 68.900 | 65.440 \| 0.283 0.461 0.256 |
| T=2 (kd1) | 22.750 | 39.590 | 70.890 | 69.960 \| 0.022 0.122 0.856 |
| T=3    | 44.670 | 64.870 | 69.110 | 65.430 \| 0.286 0.460 0.254 |
| T=3 (kd1) | 23.060 | 41.680 | 71.680 | 70.520 \| 0.012 0.153 0.835 |
| T=4    | 43.530 | 64.870 | 68.960 | 65.020 \| 0.285 0.483 0.232 |
| T=5    | 36.590 | 65.320 | 69.230 | 65.960 \| 0.174 0.581 0.244 |
| T=5 (kd1) | 21.900 | 42.630 | 71.570 | 70.270 \| 0.015 0.163 0.822 |
| T=6    | 43.020 | 66.390 | 70.040 | 66.680 \| 0.257 0.522 0.221 |
| T=6 (kd1) | 22.420 | 43.580 | 70.700 | 69.450 \| 0.012 0.170 0.818 |
| T=8    | 36.790 | 44.270 | 1.000 | 41.080 \| 0.173 0.447 0.381 |
| T=8 (kd1) | 22.650 | 42.960 | 71.780 | 70.350 \| 0.011 0.186 0.803 |
| T=10     | 35.780  | 3.470   | 51.440                                                       | 50.300 \| 0.202 0.000 0.798            |



##### Hidden Layer 32-64-64

| T\exit | 0      | 1      | 2      | Adaptive (ts0.5)            |
| ------ | ------ | ------ | ------ | --------------------------- |
| Baseline |         |         |                                                              |                                        |
| resnet |         |         | 71.270                                                       |                                        |
| Ours     |         |         |                                                              |                                        |
| T=0      | 46.060 | 65.260 | 67.990 | 62.950 \| 0.418 0.372 0.210 |
| T=2      | 46.300 | 65.440 | 69.210 | 63.750 \| 0.418 0.372 0.209 |
| T=3      | 46.200 | 65.680 | 69.110 | 63.850 \| 0.418 0.376 0.207 |
| T=4      | 46.520 | 64.760 | 67.950 | 63.200 \| 0.414 0.371 0.215 |
| T=5      | 45.040 | 65.040 | 68.860 | 63.750 \| 0.410 0.382 0.208 |
| T=6      | 47.150 | 65.170 | 67.200 | 63.340 \| 0.426 0.357 0.218 |
| T=8      | 45.450 | 64.150 | 65.580 | 61.600 \| 0.396 0.379 0.225 |
| T=10      | 23.680 | 64.650 | 63.820 | 63.930 \| 0.082 0.645 0.273 |



##### resnet_2con3

| Setting                          | Acc1   | Acc2   | Acc3   | Adaptive Acc                |
| -------------------------------- | ------ | ------ | ------ | --------------------------- |
| baseline                         |        |        |        |                             |
| resnet                           |        |        | 71.020 |                             |
| Ours                             |        |        |        |                             |
| T=0                              | 44.710 | 64.360 | 68.820 | 63.310 \| 0.389 0.391 0.220 |
| T=0 (se)                         | 45.950 | 63.810 | 68.650 | 64.180 \| 0.361 0.400 0.238 |
| T=0 (scan)                       | 47.070 | 64.660 | 69.410 | 63.450 \| 0.448 0.353 0.199 |
| T=0 (att2)                       | 45.410 | 64.450 | 69.350 | 63.180 \| 0.414 0.375 0.210 |
|                                  |        |        |        |                             |
| T=2                              | 45.410 | 63.800 | 69.060 | 63.670 \| 0.389 0.374 0.237 |
| T=2 (scan)                       | 46.590 | 64.550 | 69.030 | 63.620 \| 0.421 0.372 0.207 |
| T=2 (scan + att2, T3.0,gamma0.5) | 44.440 | 63.800 | 70.340 | 66.090\| 0.330 0.372 0.297  |
| T=5                              | 43.460 | 63.680 | 68.680 | 62.580 \| 0.383 0.385 0.232 |
| T=5 (se)                         | 45.220 | 63.570 | 68.210 | 63.700 \| 0.369 0.390 0.240 |
| T=5 (scan)                       | 47.390 | 64.250 | 69.280 | 63.540 \| 0.433 0.359 0.208 |
| T=8                              | 46.040 | 63.580 | 57.390 | 61.180 \| 0.417 0.347 0.236 |
| T=8 (se)                         | 45.420 | 62.250 | 67.790 | 63.190 \| 0.354 0.394 0.253 |
| T=8 (scan)                       | 46.300 | 64.280 | 65.910 | 61.340 \| 0.438 0.341 0.221 |
| T=10                             | 47.120 | 3.090  | 2.500  | 31.890 \|0.415 0.000 0.585  |

