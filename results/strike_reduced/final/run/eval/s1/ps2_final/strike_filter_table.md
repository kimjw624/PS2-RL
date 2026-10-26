Nominal: striker striker_rs0p1_s1 (residual on pursuit at 6 m/s); 1000 arena scenes (seed 9001); filter: discrete backup-set check with a committed plan, T = 1.5 s, chaser balls r = 3.7 m, chaser hold 0 s; chaser model guidance, hover latch True, free pursuit.

Contact = left the safe set (obstacle, floor, chaser); leaving the arena and running out of time end the mission but are not contacts.

| mode | strike | contact | obstacle | floor | chaser | out of arena | timeout | strike time [s] | kept | no-certificate episodes |
|---|---|---|---|---|---|---|---|---|---|---|
| no filter | 0.608 | 0.392 | 0.375 | 0.013 | 0.004 | 0.000 | 0.000 | 5.15 | 1.00 | 0.000 |
| filter, learned backup | 0.968 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.032 | 6.30 | 0.89 | 0.015 |
| filter, hover-LQR backup | 0.799 | 0.013 | 0.000 | 0.000 | 0.013 | 0.000 | 0.188 | 7.53 | 0.69 | 0.094 |

By scene type (strike / contact / episodes that lost the certificate):

| mode | no chaser | with chasers |
|---|---|---|
| no filter | nan / nan / nan (n = 0) | 0.608 / 0.392 / 0.000 (n = 1000) |
| filter, learned backup | nan / nan / nan (n = 0) | 0.968 / 0.000 / 0.015 (n = 1000) |
| filter, hover-LQR backup | nan / nan / nan (n = 0) | 0.799 / 0.013 / 0.094 (n = 1000) |
