Nominal: striker striker_rs0p1_s0 (residual on pursuit at 6 m/s); 1000 arena scenes (seed 9001); filter: discrete backup-set check with a committed plan, T = 1.5 s, chaser balls r = 3.7 m, chaser hold 0 s; chaser model guidance, hover latch True, free pursuit.

Contact = left the safe set (obstacle, floor, chaser); leaving the arena and running out of time end the mission but are not contacts.

| mode | strike | contact | obstacle | floor | chaser | out of arena | timeout | strike time [s] | kept | no-certificate episodes |
|---|---|---|---|---|---|---|---|---|---|---|
| no filter | 0.634 | 0.366 | 0.358 | 0.004 | 0.004 | 0.000 | 0.000 | 4.99 | 1.00 | 0.000 |
| filter, learned backup | 0.984 | 0.001 | 0.000 | 0.000 | 0.001 | 0.000 | 0.015 | 5.93 | 0.91 | 0.015 |
| filter, hover-LQR backup | 0.840 | 0.007 | 0.000 | 0.000 | 0.007 | 0.000 | 0.153 | 7.08 | 0.73 | 0.076 |

By scene type (strike / contact / episodes that lost the certificate):

| mode | no chaser | with chasers |
|---|---|---|
| no filter | nan / nan / nan (n = 0) | 0.634 / 0.366 / 0.000 (n = 1000) |
| filter, learned backup | nan / nan / nan (n = 0) | 0.984 / 0.001 / 0.015 (n = 1000) |
| filter, hover-LQR backup | nan / nan / nan (n = 0) | 0.840 / 0.007 / 0.076 (n = 1000) |
