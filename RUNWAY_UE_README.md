# 활주로 조류 퇴치 드론 + UE-bCBF (PS2-RL)

새를 쫓는 RL 정책(nominal)을 활주로 진입 금지·고도 상한 안전층(CIL)이 바람 속에서도 최소한으로 휘어 주는 예제입니다. landing UE와 같은 구조(Phase I backup → CIL → Phase II)를 그대로 따릅니다.

## 문제 설정

| 항목 | 내용 |
|---|---|
| 안전 집합 S | 활주로+보호띠 진입 금지 `y ≤ 0` (x, 즉 활주로 길이 방향은 자유) / 고도 상한 `z ≤ 10 m` |
| 바람 | `d(t) = 0.5 n sin(2π·0.05 t + φ)` m/s², 관측기 d̂ (λ=20, ē=0.02). landing UE와 동일 |
| backup 종착집합 B | "활주로 반대쪽으로 2.4 m/s로 빠지면서 고도 6 m 유지" LQR 타원체 (c_B=10.5, 고도 ±0.9 m, 속도 ±1.8 m/s) |
| B 인증 | B 안에서 v_y < 0 (활주로에서 멀어지기만 함), z ≤ 6.9 m, robust Lyapunov V+/V = 0.975 (|d| ≤ 0.5) |
| x 불변성 | 동역학·S·B·backup 모두 x에 의존하지 않음 → backup 입력은 p_x = 0 |
| 새 | 활주로 횡단(50%) / 상한 위로 상승(25%) / 무작위(25%), OU 가속으로 부드럽게 방향 변경 |
| Phase II 보상 | 새 추적만 (안전은 CIL로만 보장, 페널티 없음) |

## 결과 (CPU, 128 에피소드, 같은 초기상태·바람·새)

| | unsafe | 활주로 최대 침범 | 상한 최대 초과 | 새와 평균 거리 |
|---|---|---|---|---|
| RL 추적 정책 단독 (CIL 없음) | 86.7 % | 20.7 m | 6.5 m | 0.79 m |
| 같은 정책 + 활주로 UE-bCBF CIL | **0 %** | 0 | 0 | 5.9 m |

- Phase I backup: 회복가능 비율(tightened C_N) 0.743, test 0.732. LQR만 쓰면 0.46.
- 그림: `results/runway_ue/` (`runway_overview.png`, 3D `runway_3d*.png`, 애니메이션 `runway_episode.gif`).
- 모델: `checkpoints/runway_ue/phase1_seed0` (Phase I backup), `checkpoints/runway_ue/chaser_vanilla_seed0` (RL 추적 정책).
- **Phase II(CIL 통과 학습)는 아직 개선이 안 됩니다.** warm start 시점(추적 정책 + CIL)이 가장 좋고, 학습할수록 경계에서 멀어지며 거리가 5.8 m에서 8–9 m로 늘었습니다(15–20만 스텝). yaw-rate 보정 문제는 고쳤지만 남은 원인은 확인하지 못했습니다. 아래 3단계는 실험용입니다.

## 파일

활주로 코드는 landing UE 모듈(UE 튜브, Phase I UE 학습 루프, UE CIL의 QP/safeguard, cached Phase II)을 재사용합니다. 그래서 이 브랜치에는 landing UE 커밋이 먼저 들어 있고, 그 위에 활주로 커밋이 하나 있습니다.


- `ps2rl/envs/quadrotor_runway_config.py`: 형상·B·LQR 설정 (단일 진실 원천)
- `ps2rl/base_controller/quadrotor_retreat_dlqr.py`: retreat LQR(7-D)와 튜브용 8-D metric chart
- `ps2rl/sets/runway_sets.py`, `runway_certificate.py`: S, B, B 인증
- `ps2rl/uncertainty/runway_ue_tube.py`: UE 튜브 margin (활주로·상한·B)
- `ps2rl/phase1_sa/runway_design_region.py`, `runway_ue_sa_env.py`, `runway_ue_sa_trainer.py`: Phase I
- `ps2rl/phase1_sa/quadrotor_landing_ue_sa_trainer.py`: 공용 Phase I 업데이트. 옵션 3개를 추가했고, 기본값에서는 landing 결과가 비트 단위로 같습니다(smoke 테스트로 확인).
  - `actor_start_update`
  - `bc_weight`
  - `metric_chart` 사용
- `ps2rl/cil/quadrotor_runway_ue_bcbf.py`: 활주로 UE-bCBF CIL. QP 목적함수 가중치는 (추력, roll, pitch, yaw rate) = (1, 1, 1, 25)입니다.
- `ps2rl/envs/quadrotor_runway_bird_env.py`, `ps2rl/phase2_ps2/runway_ue_ps2_trainer.py`: 새 추적 환경과 Phase II 학습기
- `scripts/train_phase1_runway_ue.py`, `train_phase2_runway_ue.py`, `eval_runway_ue.py`, `plot_runway_3d.py`: 실행·평가·3D 그림

## 실행 (GPU)

```bash

# 0) 제 모델로 그림 다시 그리기
JAX_ENABLE_X64=1 python scripts/eval_runway_ue.py \
  --case "RL chaser alone (no CIL)=checkpoints/runway_ue/chaser_vanilla_seed0:none" \
  --case "same chaser + runway UE-bCBF CIL=checkpoints/runway_ue/chaser_vanilla_seed0:ue" \
  --episodes 128 --out outputs/runway_phase2_ue/eval_ckpt --gif
python scripts/plot_runway_3d.py --eval_dir outputs/runway_phase2_ue/eval_ckpt --gif   # 3D 그림

# 1) Phase I backup (LQR 모방으로 시작, critic 먼저 학습, LQR 쪽으로 약하게 묶음)
python scripts/train_phase1_runway_ue.py --seed 0 --total_steps 3000000 --hidden_size 128 \
  --contraction_weight 0 --start_steps 0 --actor_start_update 20000 --bc_weight 0.5 --run_name rwy_ue_seed0

# 2) RL 새 추적 정책 (nominal, CIL 없음)
JAX_ENABLE_X64=1 python scripts/train_phase2_runway_ue.py --ckpt outputs/runway_phase1_ue/rwy_ue_seed0 \
  --use_projection false --env_terminate_on_unsafe false --total_steps 3000000 --update_every 2 \
  --replay_size 1000000 --run_tag rwy_chase_vanilla_s0

# 3) (실험) Phase II: CIL 통과 학습, actor warm start + critic 새로 + actor 잠시 고정
JAX_ENABLE_X64=1 python scripts/train_phase2_runway_ue.py --ckpt outputs/runway_phase1_ue/rwy_ue_seed0 \
  --warm_start_weights outputs/runway_phase2_ue/rwy_chase_vanilla_s0/best_weights.pkl \
  --warm_start_critic false --actor_start_update 10000 --start_steps 0 --update_after 8192 \
  --actor_lr 5e-5 --critic_lr 3e-4 --alpha_lr 5e-5 --update_every 8 --rows_keep 203 \
  --env_w_wz 0.02 --total_steps 2000000 --run_tag rwy_ps2_s0
```

활주로 CIL의 행 수는 2 × 101 + 1 = 203입니다. 그래서 `--rows_keep 203`은 모든 행을 그대로 쓰는 설정입니다.
