# FLOPs

### FLOPs

In our model, "127,796,480 " in "127,796,480 + 2,560" means the flops with no feedbacks, while "2,560" means if we add 1 circles in our model, 2560 flops will be added to our model.

| Classifiers | FLOPs                | params    |
| ----------- | -------------------- | --------- |
| PCN         |                      |           |
| circles=0   | 547,230,720          | 5,212,816 |
| circles=1   | 1,039,733,760        | 9,898,192 |
| circles=2   | 1,532,236,800        | 9,898,192 |
| Ours        |                      |           |
| exit-1      | 127,796,480 + 2,560  | 128,026   |
| exit-2      | 253,628,260 + 7,880  | 626,176   |
| exit-3      | 547,234,760 + 18,320 | 5,227,750 |



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
| T=0                                | 80.96 | 90.19 | 93.09         |
| T=1                                | 83.47 | 91.05 | 92.82         |
| T=1 (val no fb)                    | 77.13 | 89.74 | 92.08         |
| T=1 (val no fb, dp0.5)             | 80.82 | 91.12 | 93.87         |
| T=1 (val no fb, dp0.5, all15clf10) | 80.24 | 91.11 | 93.98         |
| T=1 (val no fb, dp0.5, all35clf15) | 81.10 | 91.24 | 94.24         |
| T=1 (val no fb, dp0.3)             | 81.30 | 91.02 | 93.49         |
| T=2 (val no fb, dp0.5)             | 80.96 | 90.89 | 93.49         |







### CIFAR100

| Setting                            | Acc1  | Acc2  | Acc3  |
| ---------------------------------- | ----- | ----- | ----- |
| PCN                                |       |       |       |
| T=0                                |       |       | 74.69 |
| T=1                                |       |       | 76.22 |
| T=2                                |       |       | 77.25 |
| Ours                               |       |       |       |
| vanilla                            | -     | -     | -     |
| T=0                                | 50.24 | 67.82 | 72.99 |
| T=1                                | 56.91 | 68.30 | 72.14 |
| T=1 (val no fb)                    | 24.07 | 54.96 | 67.48 |
| T=1 (val no fb, dp0.5)             | 47.17 | 67.26 | 72.37 |
| T=1 (val no fb, dp0.5, ge)         | 47.49 | 67.09 | 73.48 |
| T=1 (val no fb, dp0.5, all15clf10) | -     | -     | -     |
| T=1 (val no fb, dp0.5, all35clf15) | 47.25 | 67.25 | 72.87 |
| T=1 (val no fb, dp0.3)             | -     | -     | -     |
| T=2 (val no fb, dp0.5)             | -     | -     | -     |

