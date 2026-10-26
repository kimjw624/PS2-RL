Nominal: straight-line pursuit at 6 m/s; 1000 arena scenes (seed 9001); filter: discrete backup-set check with a committed plan, T = 1.5 s, chaser balls r = 3.7 m, chaser hold 0 s; chaser model static, hover latch True, free pursuit.

Contact = left the safe set (obstacle, floor, chaser); leaving the arena and running out of time end the mission but are not contacts.

| mode | strike | contact | obstacle | floor | chaser | out of arena | timeout | strike time [s] | kept | no-certificate episodes |
|---|---|---|---|---|---|---|---|---|---|---|
| no filter | 0.425 | 0.575 | 0.574 | 0.000 | 0.001 | 0.000 | 0.000 | 5.45 | 1.00 | 0.000 |
| filter, learned backup | 0.513 | 0.384 | 0.001 | 0.000 | 0.383 | 0.100 | 0.003 | 6.21 | 0.65 | 0.545 |

By scene type (strike / contact / episodes that lost the certificate):

| mode | no chaser | with chasers |
|---|---|---|
| no filter | nan / nan / nan (n = 0) | 0.425 / 0.575 / 0.000 (n = 1000) |
| filter, learned backup | nan / nan / nan (n = 0) | 0.513 / 0.384 / 0.545 (n = 1000) |
