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

### Table 1: 전략별 해결률 (Qwen2.5-Coder-7B-Instruct, 2026-04-18)

| 전략 | Solved | 해결률 | vs one_shot |
|---|---|---|---|
| one_shot (15 candidates × 1 iter) | 33 | 12.9% | — |
| blind_retry (5 × 3 iter, 피드백 없음) | 40 | 15.7% | +2.8pp |
| **error_aware (5 × 3 iter, 유형별 피드백)** | **44** | **17.3%** | **+4.4pp** |

### 통계적 유의성 (McNemar exact)

| 비교 | A만 성공 / B만 성공 | p-value |
|------|---------------------|---------|
| error_aware vs one_shot | +18 / -7 | **0.043** ✅ |
| error_aware vs blind_retry | +12 / -8 | 0.503 ⚠️ |
| blind_retry vs one_shot | +11 / -4 | 0.119 |

반복 수리 자체는 one-shot 대비 유의미. 오류 유형 인식의 추가 효과는
test-fail 서브그룹에서 뚜렷하지만 전체 유의성은 단일 seed 한계로 경계선.

### Table 2: 초기 실패 유형별 해결률 (킬러 표)

분류 기준: **error_aware 전략의 iter 0 실패 유형** (세 전략 동일 분할로 비교).

| 초기 실패 유형 | N | one_shot | blind_retry | **error_aware** |
|---|---|---|---|---|
| 1-shot 통과 (쉬움) | 17 | 17 (100%) | 17 (100%) | 17 (100%) |
| **Test-fail** | **139** | 11 (7.9%) | 14 (10.1%) | **19 (13.7%)** |
| Parse-fail | 55 | 1 (1.8%) | 4 (7.3%) | **5 (9.1%)** |
| Compile: cannot-find-symbol | 18 | 2 (11.1%) | 2 (11.1%) | 1 (5.6%) ⚠️ |
| Compile: syntax/parse | 11 | 0 | 0 | 0 |
| Compile: type-mismatch | 3 | 2 (66.7%) | 1 (33.3%) | 1 (33.3%) |
| Compile: method-signature | 3 | 0 | 0 | 0 |
| Compile: other | 5 | 0 | 1 (20%) | 1 (20%) |
| Timeout | 2 | 0 | 1 (50%) | 0 |
| Infra-fail (Defects4J 체크아웃 실패) | 2 | 0 | 0 | 0 |
| **전체** | **255** | **33 (12.9%)** | **40 (15.7%)** | **44 (17.3%)** |

### 핵심 발견

- 가장 큰 카테고리(test-fail, 139/255)에서 error_aware가 +5 bugs (36% 상대 개선)
- parse-fail에서도 일관된 우위
- **부정 발견**: cannot-find-symbol(N=18)에서 error_aware가 오히려 열세 → symbol_feedback 개선 여지 (limitation)

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

## 에러 유형 분류 체계 (error_aware 피드백 라우팅)

각 에러 유형마다 **다른 피드백 프롬프트**를 사용해 LLM에 컨텍스트를 선별적으로 제공한다.
유형별 정량 결과는 위 Table 2를 참고.

| 에러 유형 | 피드백에 포함하는 컨텍스트 |
|---|---|
| test_fail | 실패 테스트 이름 + buggy line + "원본에서 최소 수정" 지시 |
| cannot_find_symbol | sibling methods (동일 클래스의 다른 메서드 시그니처) |
| no_valid_candidates | 파싱 실패 → 재생성 요청 (포맷 강조) |
| other_compile_fail | compiler stderr 첫 N줄 |
| method_signature | sibling method signatures |
| syntax_or_parse | compiler stderr |

## 실행 방법

### 실험 재현 (3 strategies × 255 bugs)

```bash
# 전체 실험 (순차, ~15-20시간)
BUGS=$(seq -s, 1 255)
bash run.sh --config config/ablation_1shot.yaml --bug_id_list $BUGS --max_model_len 4096
bash run.sh --config config/ablation_blind_retry.yaml --bug_id_list $BUGS --max_model_len 4096
bash run.sh --config config/ablation_error_aware.yaml --bug_id_list $BUGS --max_model_len 4096
```

### 논문 산출물 재생성

```bash
conda activate fse && python paper_artifacts/generate_paper_artifacts.py
```

출력물 (`paper_artifacts/`):
- `PAPER_SUMMARY.md` — 논문 준비 요약 (수치/통계/케이스 스터디)
- `table1_overall.tex`, `table2_breakdown.tex` — LaTeX 테이블
- `fig_breakdown.png/pdf` — Figure 1 (유형별 바차트)
- `stats_summary.txt` — McNemar 통계
- `case_study_bug153.md` — Bug 153 (linearCombination) 정성 분석

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
│   └── iterative_*_topk_20260418.json  ★ 최종 반복 수리 결과 (3 strategies)
└── paper_artifacts/            ★ 논문용 LaTeX 표 / 그래프 / 통계
```
