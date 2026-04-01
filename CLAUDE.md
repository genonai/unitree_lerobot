# unitree_IL_lerobot/ — Unitree IL Training Framework

Unitree G1용 모방학습 파이프라인. Unitree 커스텀 레이어 + HuggingFace LeRobot 서브모듈 구성. 서드파티 코드 (Unitree 제공).

## 디렉토리 구조

```
unitree_lerobot/
├── utils/
│   ├── constants.py                        # ROBOT_CONFIGS dict → RobotConfig dataclass
│   ├── convert_unitree_json_to_lerobot.py  # JSON → LeRobot Parquet+MP4 (tyro CLI)
│   ├── convert_dex3_to_inspire_1dof.py     # 28D Dex3 → 16D Inspire 1-DOF 리매핑
│   ├── convert_unitree_json_to_h5.py       # JSON → HDF5 (대안 포맷)
│   ├── convert_lerobot_to_h5.py            # LeRobot → HDF5
│   └── sort_and_rename_folders.py          # 데이터셋 폴더 정리
├── eval_robot/
│   ├── eval_g1.py              # 실제 G1 eval — closed-loop 정책 추론
│   ├── eval_g1_sim.py          # 시뮬 G1 eval — closed-loop + reward + episode 저장
│   ├── eval_g1_dataset.py      # 데이터셋 기반 eval
│   ├── replay_robot.py         # Open-loop 액션 리플레이
│   ├── make_robot.py           # 로봇 인스턴스 헬퍼 (setup_image_client, setup_robot_interface)
│   ├── robot_control/          # xr_teleoperate robot_control/ 미러
│   ├── image_server/           # eval용 카메라 스트리밍
│   └── utils/                  # sim_state_topic, rerun_visualizer, episode_writer, utils.py
└── lerobot/                    # HuggingFace LeRobot 서브모듈 (commit 0878c68)
    └── src/lerobot/
        ├── scripts/            # lerobot_train.py, lerobot_eval.py, lerobot_dataset_viz.py
        ├── policies/           # act/, diffusion_policy/, groot/, pi0/, smolvla/
        ├── datasets/           # lerobot_dataset.py, transforms.py, sampler.py
        ├── configs/            # default.py, train.py, eval.py, policies.py
        ├── processor/          # batch_processor.py, observation_processor.py
        ├── cameras/            # realsense/, opencv/
        └── robots/             # config.py (robot type registry)
test/
├── conftest.py
├── test_inspire_1dof_conversion.py
├── test_inspire_1dof_config.py
├── test_dex3_to_inspire_1dof.py
├── test_load_dataset.py, test_load_h5.py
├── test_local_push_to_hub.py
└── test_recording_e2e.py
```

## 데이터 변환 파이프라인

1. **JSON → LeRobot**: `convert_unitree_json_to_lerobot.py` (tyro CLI)
   - `--raw-dir` → `--repo-id` → `--robot_type`
   - 출력: Parquet (state/action) + MP4 (video)
2. **Dex3 → Inspire 1-DOF**: `convert_dex3_to_inspire_1dof.py`
   - 28D (arm 14 + dex3 hand 14) → 16D (arm 14 + 1DOF grip + pad)

## Robot Configs (constants.py)

`ROBOT_CONFIGS` dict에서 `RobotConfig` dataclass로 관리:

| robot_type | 차원 | 설명 |
|-----------|------|------|
| `Unitree_G1_Gripper` | 16D | 시뮬 그리퍼 (arm 14 + grip 2) |
| `Unitree_G1_Inspire_1DOF` | 16D | Inspire 1-DOF (arm 14 + grip 2) |
| `Unitree_G1_Dex3` | 28D | Dex3 (arm 14 + hand 14) |
| `Unitree_G1_Inspire` | 26D | Inspire 6-DOF (arm 14 + hand 12) |

**핵심**: Dex3 28D 데이터셋은 Inspire 16D와 직접 호환 불가 → 변환 필요.

## 학습 (Training)

```bash
cd lerobot && python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=<dataset> --policy.type=<policy>
```

지원 정책: `act` (ACT), `diffusion` (Diffusion Policy), `groot` (GR00T), `pi0`, `smolvla`

## 평가 (Evaluation)

| 스크립트 | 모드 | 설명 |
|---------|------|------|
| `eval_g1.py` | Real closed-loop | 실제 로봇에서 정책 추론 (@parser.wrap, EvalRealConfig) |
| `eval_g1_sim.py` | Sim closed-loop | 시뮬에서 정책 추론 + reward + episode 저장 |
| `eval_g1_dataset.py` | Dataset eval | 데이터셋 기반 평가 |
| `replay_robot.py` | Open-loop replay | 데이터셋 액션 그대로 재생 |

## 실행 환경

- conda env: `unitree_lerobot`
- LeRobot 서브모듈: `lerobot/` (commit 0878c68, HuggingFace)
- CLI: tyro (변환 스크립트), hydra 아님

## 테스트

```bash
cd /Users/jihun/work/IL_PhysicalAI && python -m pytest test/ -v
```

- `test_inspire_1dof_*.py` — Inspire 1-DOF 변환/설정 검증
- `test_dex3_to_inspire_1dof.py` — Dex3→Inspire 리매핑 검증
- `test_load_dataset.py` / `test_load_h5.py` — 데이터셋 로딩
- `test_recording_e2e.py` — 녹화 end-to-end

## 주의사항

- Dex3 데이터셋(28D)과 Inspire(16D) 차원을 혼용하지 않기
- 서드파티 코드이므로 직접 수정하지 않고 feature branch에서 작업
