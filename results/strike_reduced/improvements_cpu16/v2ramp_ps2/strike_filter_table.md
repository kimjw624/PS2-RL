Nominal: striker striker_rs0p1_s0 (residual on pursuit at 6 m/s); 1000 arena scenes (seed 9001), episodes [69, 181, 271, 322, 470, 480, 509, 576, 601, 752, 827, 845, 934, 949, 972, 994] only; filter: discrete backup-set check with a committed plan, T = 3.0 s, backup trained with moving chasers (h_c, hover margin 2 m) and arena walls; chaser model guidance, hover latch True, free pursuit.

Contact = left the safe set (obstacle, floor, chaser); leaving the arena and running out of time end the mission but are not contacts.

| mode | strike | contact | obstacle | floor | chaser | out of arena | timeout | strike time [s] | kept | no-certificate episodes |
|---|---|---|---|---|---|---|---|---|---|---|
| filter, learned backup | 0.938 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | 8.07 | 0.77 | 0.000 |

By scene type (strike / contact / episodes that lost the certificate):

| mode | no chaser | with chasers |
|---|---|---|
| filter, learned backup | nan / nan / nan (n = 0) | 0.938 / 0.000 / 0.000 (n = 16) |
