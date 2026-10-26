Nominal: striker striker_rs0p1_s1 (residual on pursuit at 6 m/s); 1000 arena scenes (seed 9001); filter: discrete backup-set check with a committed plan, T = 1.5 s, chaser balls r = 3.7 m, chaser hold 0 s; chaser model guidance, hover latch True, free pursuit.

Contact = left the safe set (obstacle, floor, chaser); leaving the arena and running out of time end the mission but are not contacts.

| mode | strike | contact | obstacle | floor | chaser | out of arena | timeout | strike time [s] | kept | no-certificate episodes |
|---|---|---|---|---|---|---|---|---|---|---|
| no filter | 0.748 | 0.252 | 0.204 | 0.037 | 0.011 | 0.000 | 0.000 | 5.50 | 1.00 | 0.000 |
| filter, learned backup | 0.938 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 | 6.66 | 0.90 | 0.014 |

By scene type (strike / contact / episodes that lost the certificate):

| mode | no chaser | with chasers |
|---|---|---|
| no filter | nan / nan / nan (n = 0) | 0.748 / 0.252 / 0.000 (n = 1000) |
| filter, learned backup | nan / nan / nan (n = 0) | 0.938 / 0.000 / 0.014 (n = 1000) |
