Nominal: striker striker_rs0p1_s0 (residual on pursuit at 6 m/s); 1000 arena scenes (seed 9001), episodes [5, 7, 16, 20, 56, 82, 159, 208, 222, 242, 270, 277, 283, 297, 299, 346, 368, 448, 456, 473, 480, 491, 493, 501, 530, 550, 573, 591, 617, 654, 658, 669, 675, 678, 721, 750, 770, 784, 821, 823, 828, 845, 872, 878, 888, 911, 931, 994] only; filter: discrete backup-set check with a committed plan, T = 3.0 s, backup trained with moving chasers (h_c, hover margin 2 m) and arena walls; chaser model guidance, hover latch True, free pursuit.

Contact = left the safe set (obstacle, floor, chaser); leaving the arena and running out of time end the mission but are not contacts.

| mode | strike | contact | obstacle | floor | chaser | out of arena | timeout | strike time [s] | kept | no-certificate episodes |
|---|---|---|---|---|---|---|---|---|---|---|
| filter, learned backup | 0.938 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | 8.98 | 0.76 | 0.167 |

By scene type (strike / contact / episodes that lost the certificate):

| mode | no chaser | with chasers |
|---|---|---|
| filter, learned backup | nan / nan / nan (n = 0) | 0.938 / 0.000 / 0.167 (n = 48) |
