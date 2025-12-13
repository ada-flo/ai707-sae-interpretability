# SAELens 실험 정리

- `sandbox-sae-llama-3-8b-train.ipynb`: `meta-llama/Meta-Llama-3-8B`에 SAE를 학습. SAE 훈련 결과를 `checkpoints/`, `runs/`에 저장.
- `sandbox-sae-llama-3-8b-analysis.ipynb`: 로컬 Llama-3-8B SAE 실행 결과(`runs/<timestamp>_llama3_8b`)를 불러와 희소도·재구성 검증을 재현.
- `tutorial-training-sae.ipynb`: TinyStories(tiny-stories-1L-21M)로 SAE를 학습하며 SAELens 전체 흐름을 보여주는 예제.
- `tutorial-analysis.ipynb`: 사전 학습 SAE를 로드해 희소도, 재구성, 간단한 기능 실험을 수행하는 기본 평가 튜토리얼.
- `tutorial-logit-lens.ipynb`: GPT-2 Small SAE 특징에 Logit distribution 분석을 적용하고 토큰 세트 테마 분석과 neuronpedia에서 결과 비교를 포함.

## Quickstart
현 환경은 uv 패키지매니저에 의존합니다.

- 의존성 설치(`uv`): `exp-sae-lens` 경로에서 uv를 이용해서 가상환경 생성 `uv venv`, 이후 `uv sync` (Python 3.10).
- 노트북 실행: `uv run jupyter lab` (또는 IDE에서 열어서 .venv로 커널 잡아주고) 후 원하는 `.ipynb` 열기.
- Hugging Face 토큰: Meta Llama 3 등 접근 제한 모델이 필요하면 `huggingface-cli login`.

## Llama-3-8B 실행 가이드

- 하드웨어: 단일 ~48GB GPU(A6000)에 맞춘 설정. OOM 시 `context_size`, `d_sae`, `train_batch_size_tokens`를 줄이기.
- 학습: `sandbox-sae-llama-3-8b-train.ipynb`에서 `device`/`hook_name`/`d_sae` 등을 조정 후 설정 셀 실행(기본값: `blocks.15.hook_resid_pre`, `d_sae=16384`, `context_size=256`, 데이터셋 `monology/pile-uncopyrighted`, BF16). 체크포인트는 `checkpoints/<timestamp>_llama3_8b`, 최종 SAE와 로그는 `runs/<timestamp>_llama3_8b`.
- 데이터/모델 다운로드: 첫 실행 시 Pile-uncopyrighted를 스트리밍 및 캐싱하고 `meta-llama/Meta-Llama-3-8B`를 받음. HF 토큰에 Llama 3 접근 권한이 있는지 확인.
- 분석: 학습 후 `sandbox-sae-llama-3-8b-analysis.ipynb`에서 `run_dir`를 `runs/` 내 결과로 지정해 L0 희소도, 재구성 vs. ablation, 간단한 프롬프트 동작을 확인.
