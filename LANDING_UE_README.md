# Landing + 외란 (UE-bCBF): 기존 코드 구조에 맞춘 패치

## 1. 적용

```bash
cd ~/PS2-RL
git checkout main && git pull                     # 3bac818 이상 (친구분 landing Phase I / CIL)
git checkout -b landing-ue
git am -3 /path/to/landing_ue_bcbf/0001-*.patch   # UE 모듈 (새 파일만)
git am -3 /path/to/landing_ue_bcbf/0002-*.patch   # repo trainer에 연결 (기존 파일은 ps2_trainer_core.py 하나)

# 학습된 모델
mkdir -p outputs/landing_phase1_ue outputs/landing_phase2_ue
cp -r /path/to/landing_ue_bcbf/artifacts/phase1/ue_floor_rec10_td3_h128_seed0 outputs/landing_phase1_ue/
cp -r /path/to/landing_ue_bcbf/artifacts/phase2/land_ue_p2_res_s0           outputs/landing_phase2_ue/
```
- 예전 landing 패키지의 x64 한 줄 수정이 이미 들어가 있어도 `-3`로 충돌 없이 합쳐집니다 (확인함).
- git을 안 쓰면 `files/` 아래를 같은 경로로 덮어쓰면 됩니다.

## 2. 기존 구조에서 어디에 들어가나

### Phase I — 친구분 nominal landing과 같은 층에 나란히

```
scripts/train_phase1_landing.py     ─> phase1_sa/quadrotor_landing_sa_trainer.py     ─┐
scripts/train_phase1_landing_ue.py  ─> phase1_sa/quadrotor_landing_ue_sa_trainer.py  ─┴─> phase1_sa/sa_trainer_core.py (그대로)
                                       ├ phase1_sa/quadrotor_landing_ue_sa_env.py   (= quadrotor_landing_sa_env.py의 UE판)
                                       └ uncertainty/landing_ue_tube.py             (quadrotor_disturbance*.py 옆)
```
- `train_phase1_landing_ue.py`는 `train_phase1_landing.py`의 인자 파서·인증 함수를 import해서 씀 → 인자 이름 동일 + `--ue_*`
- core는 손대지 않음: 학습 루프·replay·curriculum·eval hook은 공용, UE 쪽은 update 함수(액터 입력 13-D + 수축 정규화)와
  수집 스텝(튜브 성장률 계산)만 자기 것

### Phase II — 기존 `train_phase2_landing.py`와 같은 "swap" 방식

```
scripts/train_phase2_landing.py    (기존, nominal)   ┐
scripts/train_phase2_landing_ue.py (새, UE)          ┴─ swap ─> quadrotor_ps2_entry.main ─> quadrotor_ps2_trainer ─> ps2_trainer_core
```

| swap 지점 | nominal landing (기존) | UE landing (새) |
|---|---|---|
| `entry.QuadrotorBCBFConfig` | `landing_bcbf_config_from_checkpoint` | `ue_bcbf_config_from_checkpoint` |
| `entry/trainer.QuadrotorBackupCBFProjector` | `QuadrotorLandingBackupCBFProjector` | `QuadrotorLandingUEBackupCBFProjector` |
| `entry/trainer.build_quadrotor_env` | `build_quadrotor_landing_env` | `build_quadrotor_landing_ue_env` |
| `trainer._BINDING` | `LANDING_BINDING` | `make_landing_ue_binding(residual=...)` |
| `trainer._PHYS_DIM` | 10 | **13** (x + d̂) |
| `trainer._validate_learned_backup_policy_compatibility` | 10-D backup 검사 | 13-D (x, d̂) backup 검사 |
| `trainer._init_state` | — | residual이면 actor mean head를 0으로 (시작 = tracker) |
| `entry._build_metric_logger` | `landing_reward.json` 기록 | `landing_ue.json` 기록 |

새 모듈이 기존 모듈 옆에 1:1로 놓입니다:

| 기존 | 새 (UE) | 같은 인터페이스 |
|---|---|---|
| `cil/quadrotor_landing_backup_cbf.py` | `cil/quadrotor_landing_ue_bcbf.py` | `solve_backup_cbf_qp_batch(_with_info)`, `backup_policy_batch`, Projector 클래스 |
| `phase2_ps2/landing_ps2_binding.py` | `phase2_ps2/landing_ue_ps2_binding.py` | `PS2SystemBinding` |
| `envs/quadrotor_landing_env.py` | `envs/quadrotor_landing_ue_env.py` | `QuadrotorEnvFns` + `QuadrotorStepInfo` 필드 전부 |

관측 순서 (trainer가 `obs[..., :phys_dim]`로 CIL 입력을 자르므로 d̂를 x 바로 뒤에 둠):

| x (10) | d̂ (3) | ref state (10) | ref ω (3) | t, sin, cos | u_nom (4, residual일 때) |
|---|---|---|---|---|---|

`next_obs[:10]` = x는 그대로라 trainer의 `_evaluate_policy`·궤적 그림이 그대로 동작합니다.

### 기존 파일 수정은 `ps2_trainer_core.py` 하나

- `PS2SystemBinding`에 optional 필드 2개 추가:
  - `actor_bounds_fn` — 액터의 tanh 박스 (residual 박스)
  - `reference_action_fn(obs, a)` — CIL에 넣을 u_ref (residual: u_nom + a)
- update / action / batched-action 함수가 이 두 개를 받도록 인자 추가. **둘 다 None이면 원래 코드와 결과가 비트 단위로 같음**
  (main 버전과 같은 입력으로 update·action 출력 122개 텐서 비교, 차이 0).
- float64 QP일 때 metric 누적 dtype 한 줄 (예전 패키지와 같은 줄).

### 빠른 CPU 경로 (선택)

`scripts/train_phase2_landing_ue_cached.py` + `phase2_ps2/landing_ue_ps2_trainer.py`: 같은 SAC-through-CIL인데
CIL 행을 replay에 저장해 update에서 QP만 풂 (CPU에서 ~10배 빠름). 배포 모델은 이걸로 학습했고, repo 경로와
관측·액터 구조가 같아서 `eval_phase2_landing_ue.py`는 둘 다 읽습니다. GPU가 있으면 repo 경로를 쓰면 됩니다.

## 3. 실행

```bash
# Phase I (CPU 2코어 93분)
python scripts/train_phase1_landing_ue.py --config_json docker_batch/configs/landing_cone45_r0p3.json \
  --floor_constraint true --region_edge_floor_prob 0.3 --recovery_rate_cone 10 --recovery_rate_floor 10 \
  --seed 0 --total_steps 5000000 --hidden_size 128 --contraction_weight 0.01 --contraction_target 1.0 \
  --eval_every 250000 --log_every 50000 --run_name ue_floor_rec10_td3_h128_seed0

# Phase II — repo trainer (기존 train_phase2_landing.py와 같은 방식, entry 인자 그대로 사용)
JAX_ENABLE_X64=1 python scripts/train_phase2_landing_ue.py --qp_float64 --seed 0 --total_steps 5000000 \
  --landing_ue_ckpt outputs/landing_phase1_ue/ue_floor_rec10_td3_h128_seed0 \
  --batch_size 256 --actor_lr 3e-4 --critic_lr 3e-4 --alpha_lr 3e-4 --min_alpha 1e-2 --q_clip_abs 1e4 \
  --start_steps 10000 --update_after 5000 --eval_every 250000 --eval_episodes 16 \
  --save_final_weights --run_tag land_ue_p2_res_s0
#   --no_residual : 순수 정책,  --alpha_floor / --res_thrust / --res_rate / --w_xy ... : UE·보상 설정

# Phase II — cached (CPU, 배포 모델 설정 그대로, 7.3시간)
JAX_ENABLE_X64=1 python scripts/train_phase2_landing_ue_cached.py \
  --ckpt outputs/landing_phase1_ue/ue_floor_rec10_td3_h128_seed0 --seed 0 --total_steps 5000000 --run_tag land_ue_p2_res_s0

# 평가 (어느 쪽 run이든)
JAX_ENABLE_X64=1 python scripts/eval_phase2_landing_ue.py --run outputs/landing_phase2_ue/land_ue_p2_res_s0 --episodes 128
```
- repo 경로 출력: `outputs/landing_phase2_ue/<run_tag>/` 에 기존과 같은 `best_weights.pkl, configs.json, metrics.jsonl,
  history.npz, best_trajectory.png` + `landing_ue.json`.
- `JAX_ENABLE_X64=1 --qp_float64`: QP만 float64 (float32면 QP 실패 ~2%). 네트워크·버퍼는 float32.
- 두 경로 모두 CPU smoke test 통과 (repo 경로: 512 스텝 학습 → eval → best_trajectory.png까지).

## 4. 수식 요약

- 외란 ẋ = f + g u + E_d d, ‖d‖ ≤ 0.5, ‖ḋ‖ ≤ 2π·0.05·0.5;  observer d̂ = λ(v − ξ), λ = 20, ‖d − d̂‖ ≤ ē = 0.02
- backup flow: d̂ 고정 rollout,  ‖d(t+τ) − d̂(t)‖ ≤ q(τ) = ē + min(δ_v τ, 2δ_d)
- 튜브 (LQR P-metric): s_{k+1} = ‖F_k‖_P s_k + dt·γ·q(τ_k);  마진 m_cone = 0.275 s̃, m_floor = 0.173 s̃, m_B = 2√c_B s̃ + s̃² (s̃ = 1.2 s)
- Phase I: 실패 h < m(s), 도달 V ≤ c_B − m_B(s);  critic (x, d̂, τ/T, s), actor (x, d̂);  수축 정규화 0.01·relu(log‖F‖_P/dt − 1)
- Phase II 행: −∇h Φ g u ≤ α(h − m) + ∇h(Φ(f + E_d d̂) − f_cl) − ē‖∇h(Φ E_d + λΘ)‖  (마지막 항 = UE-bCBF observer 오차 항)

## 5. 결과 (배포 모델 = cached trainer로 학습)

### Phase I (5M 스텝, test 세트, 같은 상태·같은 d̂·같은 튜브로 채점)

| backup | UE 기준 (튜브 축소 + d̂) | nominal 기준 (d̂ = 0, 축소 없음) |
|---|---|---|
| **UE backup (이번 학습)** | **0.417** (general 0.713 / edge 0.222 / shell 1.000) | 0.423 |
| 친구분 nominal backup | 0.287 (general 0.475 / edge 0.105 / shell 1.000) | 0.484 |

- 외란을 넣어도 UE backup의 회복 가능 영역은 거의 줄지 않음 (0.423 → 0.417). nominal backup은 0.484 → 0.287.
- 대가: 외란이 없을 때는 nominal backup보다 약간 작음 (0.423 vs 0.484; 128 hidden + 수축 정규화 영향 포함).
- B robust invariance: |d| ≤ 0.5에서 최악 V⁺/V = 0.9745 < 1.
- 그림: `artifacts/phase1/ue_floor_rec10_td3_h128_seed0/phase1_ue_curves.png`

### Phase II (5M 스텝)

학습: 5,000,704 스텝, CPU 2코어 7.3시간, best = 3.5M 스텝 (eval return 기준).
학습 전 구간에서 unsafe 에피소드 0, QP fallback 0, slack > 1e-3 비율 최대 0.03%, 캐싱한 32개 행이 전체 QP와 일치한 비율 ≥ 99.997%.

평가: 처음 보는 시작점·외란 128 에피소드 (`--seed 2026`), 같은 policy, 같은 외란, 필터만 바꿈

| 실행 | unsafe | 착륙 | 착륙 후 패드 중심 거리 | 착륙 속도 | return |
|---|---|---|---|---|---|
| **policy + UE 필터 (학습한 것)** | **0 %** (최소 h_cone +0.005 m, 최소 z +0.009 m) | **100 %** | 0.22 m | 0.16 m/s | **−203.5** |
| 같은 policy + 친구분 nominal 필터 | 9.4 % (모두 바닥 침범, 최대 0.9 mm) | 89 % | 0.17 m | 0.30 m/s | −222.1 |
| 같은 policy, 필터 없음 | 100 % (0.6 s 안에 콘 이탈) | 0 % | — | — | — |
| tracker만 + UE 필터 (Phase II 시작점) | 0 % | 100 % | 0.007 m | 0.01 m/s | −235.6 |

- 착륙 = z < 0.1 m, 패드 반경 0.3 m 안, |v| < 0.5 m/s, 안전 유지.
- nominal 필터는 외란을 모르므로 착지 직전 아래로 미는 외란에 바닥(z ≥ 0) 제약을 아주 조금 넘습니다. UE 필터는 observer 오차 항과 튜브 마진으로 이를 막습니다.
- RL이 tracker보다 나아진 부분: 고도 추종 비용 175 → 107 (−39 %). 대신 수평 오차는 조금 커져 패드 중심이 아닌 반경 0.22 m 지점에 착지 —
  보상이 "벽을 따라 미끄러져 내려가기"를 위해 고도를 수평보다 우선하도록 설계되어 있기 때문입니다 (xy 가중치 0.5, 고도 1.0/0.05 m).
  중심 착지가 중요하면 `--w_xy` (repo 경로) / `--env_w_xy` (cached)를 올리세요.
- 그림: `artifacts/phase2/land_ue_p2_res_s0/eval/landing_ue_eval.png` (옆에서 본 궤적·최소 barrier 값), `phase2_ue_curves.png` (학습 곡선).

## 6. 한계 (솔직하게)

- **형식적 보장은 아님.** 튜브는 1차 근사 + 경험적 1.2배, 마진의 x 의존성은 미분하지 않음 — 친구분 `quadrotor_ue_bcbf_experimental`과 같은 수준.
- **외란 크기**: 0.5 m/s², 0.05 Hz에서 실현 가능. 같은 진폭에 0.2 Hz면 T = 2 s 끝의 base 마진이 c_B(12)를 넘어 B가 비게 됨 (계산 확인).
- **Phase II는 residual 학습**: u_ref = u_nom + u_res (u_nom = d̂ 보상 포함 기하 tracker, residual head는 0으로 초기화).
  순수 SAC 정책(repo 경로 `--no_residual`, cached `--env_nominal_controller none`)도 같은 코드로 돌지만, CPU 예산(업데이트 1/16)에서는 35만 스텝까지 return −4800
  (tracker −223)로 학습이 안 돼서 중단했습니다 (`artifacts/phase2/pure_sac_aborted_350k.log`). 예전 nominal landing에서도 수백만 스텝이
  필요했던 것과 같은 현상이라, GPU에서 update 비율을 올려(`--update_every 8`) 다시 해볼 만합니다.
  residual 구조는 SITL에서 u = P_CIL(u_nom + u_res)로 쓰려던 것과 같은 형태라 그대로 이어집니다.
- residual 학습 초반(25만 스텝)에 return이 −766까지 떨어졌다가 회복했습니다. best checkpoint는 eval return으로 고르므로 배포 모델에는 영향 없음.
- **CPU 때문에 바꾼 것** (배포 모델): Phase I backup 128 hidden (친구분 256), Phase II update 1/16, target action projection 끔.
  repo 경로는 entry 기본값(update 1/8)을 그대로 쓰고, GPU면 `--project_target_actions`도 켤 수 있습니다.
- repo 경로로는 smoke test(512 스텝)까지만 돌렸습니다. 5M 학습 결과는 cached 경로의 것입니다.
- observer는 착륙 구간 시작 전에 수렴해 있다고 가정 (‖d − d̂‖ ≤ ē/2로 시작).
- 시뮬레이션 모델 기준 (Euler, body-rate 즉시 추종). PX4 rate loop 지연은 모델에 없음.
