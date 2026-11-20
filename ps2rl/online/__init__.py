"""Obstacle-only online safety filter (the strike method for a Tarot 450-class vehicle and a companion computer).

numpy only (no JAX on the vehicle):
  model    plant, hover LQR, safe / base set and the backup's observation for tall cylinders (twins of
           ps2rl.envs.strike_scene with n_chaser_slots = 0), and the learned backup's actor
  filter   the discrete backup filter: blends of the nominal with a backup family, first-hit certificate, committed plan,
           hover latch, decisions every P control steps
  nominal  go-to-goal velocity loop with a look-ahead heading search around the known cylinders
  sim      closed-loop episodes with a realistic plant (rate / thrust lag, delay, gain error, wind, noise) and range-limited
           obstacle sensing, batched over episodes
"""
