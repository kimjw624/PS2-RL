Nominal: striker striker_rs0p1_s0 (residual on pursuit at 6 m/s); pursuit: heading search (look-ahead 6 m, margin 1 m, chasers 3 m, heading weight 0, wall margin 2 m, straight within 3 m); 1000 arena scenes (seed 9001), episodes [5, 7, 16, 20, 56, 82, 159, 208, 222, 242, 270, 277, 283, 297, 299, 346, 368, 448, 456, 473, 480, 491, 493, 501, 530, 550, 573, 591, 617, 654, 658, 669, 675, 678, 721, 750, 770, 784, 821, 823, 828, 845, 872, 878, 888, 911, 931, 994] only; filter: discrete backup-set check with a committed plan, T = 1.5 s, chaser balls r = 3.7 m, chaser hold 0 s; chaser model guidance, hover latch True, free pursuit.

Contact = left the safe set (obstacle, floor, chaser); leaving the arena and running out of time end the mission but are not contacts.

| mode | strike | contact | obstacle | floor | chaser | out of arena | timeout | strike time [s] | kept | no-certificate episodes |
|---|---|---|---|---|---|---|---|---|---|---|
| filter, learned backup | 0.979 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.021 | 7.21 | 0.92 | 0.021 |

By scene type (strike / contact / episodes that lost the certificate):

| mode | no chaser | with chasers |
|---|---|---|
| filter, learned backup | nan / nan / nan (n = 0) | 0.979 / 0.000 / 0.021 (n = 48) |
