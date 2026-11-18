Nominal: striker striker_rs0p1_s1 (residual on pursuit at 6 m/s); pursuit: heading search (look-ahead 6 m, margin 1 m, chasers 3 m, heading weight 0, wall margin 2 m, straight within 3 m); 1000 arena scenes (seed 9001); filter: discrete backup-set check with a committed plan, T = 1.5 s, chaser balls r = 3.7 m, chaser hold 0 s; chaser model guidance, hover latch True, free pursuit.

Contact = left the safe set (obstacle, floor, chaser); leaving the arena and running out of time end the mission but are not contacts.

| mode | strike | contact | obstacle | floor | chaser | out of arena | timeout | strike time [s] | kept | no-certificate episodes |
|---|---|---|---|---|---|---|---|---|---|---|
| filter, learned backup | 0.993 | 0.003 | 0.000 | 0.000 | 0.003 | 0.000 | 0.004 | 6.41 | 0.96 | 0.015 |

By scene type (strike / contact / episodes that lost the certificate):

| mode | no chaser | with chasers |
|---|---|---|
| filter, learned backup | nan / nan / nan (n = 0) | 0.993 / 0.003 / 0.015 (n = 1000) |
