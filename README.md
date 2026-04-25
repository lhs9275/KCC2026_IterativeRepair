# KCC2026: 소형 언어 모델의 반복적 프로그램 수리를 위한 오류 유형 인식 피드백 기법

> **Error-Type-Aware Iterative Feedback for LLM-Based Program Repair with Small Language Models**

## 연구 동기

LLM 기반 자동 프로그램 수리(APR)는 대부분 1회 생성(one-shot)에 의존한다.
GPT-4 같은 대형 모델은 에러 메시지를 통째로 넘겨도 잘 이해하지만,
**7B급 소형 모델**은 긴 에러 로그를 제대로 활용하지 못한다.

## 핵심 아이디어

1. **오류 유형별 맞춤 피드백** — 컴파일/테스트 에러를 7가지로 분류하고, 유형에 필요한 컨텍스트만 선별 제공
2. **Fragment-aware merge** — 소형 LLM이 메서드 전체 대신 수정 부분만 출력하는 경향 → brace-balanced merge로 원본에 자동 삽입

## 실험 결과 (Defects4J 255 bugs)

### Table 2: 전략별 해결률

| 전략 | Solved | 해결률 | vs one_shot |
|---|---|---|---|
| one_shot (15 candidates × 1 iter) | 25 | 9.9% | — |
| blind_retry (5 × 3 iter, 피드백 없음) | 23 | 9.3% | -0.6pp |
| **error_aware (5 × 3 iter, 유형별 피드백)** | **58** | **23.7%** | **+13.8pp (2.4×)** |

### 핵심 발견

- **blind retry는 one_shot보다 못함** → 에러 정보 없는 재시도는 역효과
- **error_aware만이 반복의 가치를 실현** → 2.4배 향상
- **error_aware ⊃ one_shot ∪ blind_retry** → 다른 전략이 푼 모든 bug를 포함 (superset)
- **58개 중 43개(74%)가 iter 2/3에서 recovery** → 반복 피드백의 효과 입증
- **test_feedback이 40.5% recovery rate** → 가장 효과적인 피드백 유형

### Figure 3: Iteration별 해결 수

```
iter 1: +15  (누적: 15)
iter 2: +32  (누적: 47)   ← 가장 큰 기여
iter 3: +11  (누적: 58)
```

### Table 3: 프로젝트별 해결률

| Project | Total | one_shot | blind | error_aware |
|---|---|---|---|---|
| closure-compiler | 90 | 6% | 3% | **11%** |
| commons-math | 70 | 14% | 13% | **33%** |
| commons-lang | 37 | 11% | 14% | **19%** |
| jfreechart | 16 | 25% | 25% | **44%** |
| joda-time | 16 | 0% | 6% | **19%** |
| mockito | 16 | 12% | 6% | **50%** |

## 파이프라인 실행 순서

```
0.EnrichContext.py         버그 컨텍스트 보강 (imports, fields, siblings 추출)
        ↓
1.ASTReasonAnalyzer.py     AST 기반 버그 원인 분석
        ↓
2.Bm25.py                 BM25 유사 코드 검색
        ↓
3.GenerateBugfixPrompt.py  수리 프롬프트 생성
        ↓
4.PlanAgent.py             Plan Agent 수리 계획 생성
        ↓
5.IterativeRepair.py       ★ 반복 수리 실행 (본 연구의 핵심)
        ↓
6.AnalyzeResults.py        결과 분석 (기본)
        ↓
7.PaperAnalysis.py         ★ 논문용 분석 & 그래프 생성
```

- Stage 0~4: 기존 ICSE 파이프라인 (결과 생성 완료)
- **Stage 5~7: 본 연구에서 새로 추가한 부분**

## 반복 수리 루프 (Stage 5 상세)

```
┌─────────────────────────────────────────────┐
│  Original Prompt (Stage 4 결과)              │
└──────────────┬──────────────────────────────┘
               ▼
┌──────────────────────────┐
│  LLM 패치 후보 생성 (N개) │ ◄── temperature 점진 증가
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│  Fragment-Aware Merge     │ ◄── 소형 LLM fragment → 완전 메서드
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│  컴파일 & 테스트 실행      │
└──────────────┬───────────┘
               ▼
          ┌─ Pass ──► 수리 성공 (종료)
          │
          └─ Fail ──► 에러 유형 분류
                        │
                        ▼
               ┌────────────────┐
               │ 유형별 맞춤      │
               │ 피드백 프롬프트   │
               │ (짧고 직접적)    │
               └───────┬────────┘
                       │
                       ▼
                  다음 iteration으로 (최대 3회)
```

## 에러 유형 분류 체계

| 에러 유형 | 피드백에 포함하는 컨텍스트 | Recovery Rate |
|---|---|---|
| test_fail | buggy line location + "최소 수정" | TBD (재실험 중) |
| cannot_find_symbol | sibling methods (5개) | 23.1% |
| no_valid_candidates | (재생성) | 12.0% |
| other_compile_fail | compiler stderr | 7.7% |
| method_signature | sibling signatures | 6.7% |
| syntax_or_parse | compiler stderr | 1.3% |

## 실행 방법

### 실험 재현 (3 strategies × 255 bugs)

```bash
# 전체 실험 (순차, ~15-20시간)
BUGS=$(seq -s, 1 255)
bash run.sh --config config/ablation_1shot.yaml --bug_id_list $BUGS --max_model_len 4096
bash run.sh --config config/ablation_blind_retry.yaml --bug_id_list $BUGS --max_model_len 4096
bash run.sh --config config/ablation_error_aware.yaml --bug_id_list $BUGS --max_model_len 4096
```

### 논문 분석 & 그래프 생성

```bash
python 7.PaperAnalysis.py --output_dir paper_figures
```

출력물:
- `paper_figures/table2_strategy.csv` — Table 2 (전략 비교)
- `paper_figures/table3_project.csv` — Table 3 (프로젝트별)
- `paper_figures/fig3_iter_distribution.png` — Figure 3 (iter별 해결 수)
- `paper_figures/fig4_error_recovery.png` — Figure 4 (에러 유형별 recovery)
- `paper_figures/fig5_project_breakdown.png` — Figure 5 (프로젝트별 비교)
- `paper_figures/summary.json` — 전체 통계 JSON

### 소규모 테스트

```bash
# 버그 10개로 빠른 검증
bash run.sh --config config/ablation_error_aware.yaml --bug_id_list 1,2,3,4,5,6,7,8,9,10
```

## 환경 요구사항

- GPU: VRAM 28GB+ (Qwen 7B + vLLM)
- Java 11 (Defects4J 요구, `run.sh`에서 자동 설정)
- Python 3.10+, vLLM, transformers, torch (conda env: `kcc`)
- Defects4J v3 (초기화 필요: `cd ICSE/defects4j && bash init.sh`)

## 디렉토리 구조

```
KCC2026_IterativeRepair/
├── 0~4.*.py                    ICSE 파이프라인 (Stage 0-4)
├── 5.IterativeRepair.py        ★ 반복 수리 실행
├── 6.AnalyzeResults.py         기본 결과 분석
├── 7.PaperAnalysis.py          ★ 논문용 분석 & 그래프
├── run.sh                      환경설정 + 실행 스크립트
├── llm_backend.py              LLM 백엔드 (vLLM/HF)
├── config/                     ablation 실험 설정
│   ├── ablation_1shot.yaml
│   ├── ablation_blind_retry.yaml
│   └── ablation_error_aware.yaml
├── core/                       핵심 로직
│   ├── iterative_repair.py     반복 루프 컨트롤러
│   ├── error_classifier.py     에러 유형 분류기
│   ├── feedback_prompt.py      유형별 피드백 프롬프트 (v2)
│   └── patch_generator.py      패치 생성 + fragment merge
├── evaluation/                 평가 모듈
│   └── eval_iterative.py       compile/test 실행
├── icse_lib/                   ICSE 파이프라인 의존성
├── Results/                    실험 결과
│   ├── 1~5/                    Stage 0-5 결과
│   └── iterative_*.json        ★ 반복 수리 실험 결과
└── paper_figures/              ★ 논문용 그래프/CSV
```
