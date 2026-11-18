Nominal: striker striker_rs0p1_s0 (residual on pursuit at 6 m/s); 1000 arena scenes (seed 9001); filter: discrete backup-set check with a committed plan, T = 3.0 s, backup trained with moving chasers (h_c, hover margin 2 m) and arena walls; chaser model guidance, hover latch True, free pursuit.

Contact = left the safe set (obstacle, floor, chaser); leaving the arena and running out of time end the mission but are not contacts.

| mode | strike | contact | obstacle | floor | chaser | out of arena | timeout | strike time [s] | kept | no-certificate episodes |
|---|---|---|---|---|---|---|---|---|---|---|
| filter, learned backup | 0.970 | 0.004 | 0.000 | 0.000 | 0.004 | 0.000 | 0.026 | 6.01 | 0.90 | 0.029 |

By scene type (strike / contact / episodes that lost the certificate):

| mode | no chaser | with chasers |
|---|---|---|
| filter, learned backup | nan / nan / nan (n = 0) | 0.970 / 0.004 / 0.029 (n = 1000) |
