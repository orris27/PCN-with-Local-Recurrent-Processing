# FLOPs

| Classifiers | FLOPs             | params  |
| ----------- | ----------------- | ------- |
| 1           | 127796480 + 2560  | 128026  |
| 2           | 253628260 + 7880  | 626176  |
| 3           | 547234760 + 18320 | 5227750 |





| Setting                     | Acc1  | Acc2  | Acc3          |
| --------------------------- | ----- | ----- | ------------- |
| PCN                         |       |       |               |
| T=0                         |       |       | 94.32 (94.77) |
| T=1                         |       |       | 94.27 (94.35) |
| T=2                         |       |       | 94.61         |
| Ours                        |       |       |               |
| T=0                         | 80.96 | 90.19 | 93.09         |
| T=1                         | 83.47 | 91.05 | 92.82         |
| T=1 (val no fb)             | 77.13 | 89.74 | 92.08         |
| T=1 (val no fb, dropout0.5) | 80.82 | 91.12 | 93.87         |
| T=1 (val no fb, dropout0.3) | 81.30 | 91.02 | 93.49         |
| T=2 (val no fb, dropout0.5) | 80.96 | 90.89 | 93.49         |

