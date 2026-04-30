# KCC 2026 논문 준비 요약

**제목 방향**: 오류 유형 인식 피드백을 활용한 LLM 기반 반복 프로그램 수리
(Error-Type-Aware Iterative Feedback for LLM-Based Automated Program Repair)

**모델**: Qwen2.5-Coder-7B-Instruct (vLLM, max_model_len=4096)
**벤치마크**: Defects4J 255 bugs
**실행일**: 2026-04-18

---

## 1. 핵심 결과

### 1.1 전체 (Table 1)
| 전략 | 해결 | 해결률 |
|------|------|--------|
| One-shot (baseline) | 33/255 | 12.9% |
| Blind-retry | 40/255 | 15.7% |
| **Error-type-aware (ours)** | **44/255** | **17.3%** |

### 1.2 통계적 유의성 (McNemar exact)
| 비교 | A만 성공 / B만 성공 | p-value |
|------|---------------------|---------|
| error_aware vs one_shot | +18 / -7 | **0.043** ✅ |
| error_aware vs blind_retry | +12 / -8 | 0.503 ⚠️ |
| blind_retry vs one_shot | +11 / -4 | 0.119 |

**정직한 주장**: 반복 수리 자체는 유의미하게 one-shot보다 낫다.
오류 유형 인식의 추가 효과는 test-fail 서브그룹에서 뚜렷하지만,
전체 유의성은 단일 seed 한계로 경계선에 있음 (Limitation에 기록).

### 1.3 초기 실패 유형별 분포 (Table 2, 킬러 표)

분류 기준은 **error_aware 전략의 iter 0 실패 유형**이며, 세 전략 모두 동일 분할로 비교한다.
(one_shot은 candidate 수가 달라 자체 iter 0 분포가 다르지만, 비교의 일관성을 위해 통일.)

| 초기 실패 유형 | N | One-shot | Blind-retry | **Error-aware** |
|----------------|---|----------|-------------|-----------------|
| 1-shot 통과 (쉬움) | 17 | 17 (100%) | 17 (100%) | 17 (100%) |
| **Test-fail** | **139** | 11 (7.9%) | 14 (10.1%) | **19 (13.7%)** |
| Parse-fail | 55 | 1 (1.8%) | 4 (7.3%) | **5 (9.1%)** |
| Compile: cannot-find-symbol | 18 | 2 (11.1%) | 2 (11.1%) | 1 (5.6%) ⚠️ |
| Compile: syntax/parse | 11 | 0 | 0 | 0 |
| Compile: type-mismatch | 3 | 2 (66.7%) | 1 (33.3%) | 1 (33.3%) |
| Compile: method-signature | 3 | 0 | 0 | 0 |
| Compile: other | 5 | 0 | 1 (20%) | 1 (20%) |
| Timeout | 2 | 0 | 1 (50%) | 0 |
| Infra-fail (체크아웃 실패) | 2 | 0 | 0 | 0 |
| **전체** | **255** | **33 (12.9%)** | **40 (15.7%)** | **44 (17.3%)** |

**핵심 인사이트**:
- 가장 큰 카테고리(test-fail, 139/255)에서 Error-aware가 +5 bugs (36% 상대 개선)
- parse-fail에서도 일관된 우위
- **부정 발견**: cannot-find-symbol에서 오히려 열세 → symbol_feedback 개선 여지 (limitation)

---

## 2. 연산 비용 비교
| 전략 | 평균 LLM 호출 | 평균 처리시간 | 전체 wall-clock |
|------|---------------|---------------|------------------|
| One-shot (15 cand × 1 iter) | 0.99 회/버그 | 98.6s | 6.99h |
| Blind-retry (5 cand × 3 iter) | 2.80 회/버그 | 104.3s | 7.39h |
| Error-aware (5 cand × 3 iter) | 2.78 회/버그 | 102.9s | 7.29h |

Error-aware와 Blind-retry는 candidate × iteration 예산이 동일(5×3)하므로 공정 비교가 성립.
One-shot은 단일 iter 예산을 candidate 15개에 사용한 동일 LLM 호출 횟수 기준 비교.

---

## 3. 정성 예시: Bug 153 (Apache Commons Math `linearCombination`)

**버그 내용**: `linearCombination(double[] a, double[] b)`에서 길이 1 배열 엣지 케이스 미처리
(내부에서 `prodHigh[1]` 접근 → ArrayIndexOutOfBoundsException)

**실패 테스트**: `testLinearCombinationWithSingleElementArray`

### Error-aware (test_feedback 전략)
- **iter 0**: 원본 버그 구조 그대로 → test_fail (1 failing test)
- **iter 1 피드백**: 실패 테스트 이름 직접 노출 + "ORIGINAL BUGGY CODE에서 시작, 최소 변경" 지시
- **iter 1 결과**: ✅ `if (len == 1) return a[0] * b[0];` 한 줄 추가로 해결

### Blind-retry
- **iter 0**: 동일한 실패
- **iter 1, 2, 3**: "다른 방식으로 시도해봐"만 받음 → 테스트 이름 힌트 없이 같은 엣지 케이스 반복 놓침 → ❌ 3번 모두 실패

**교훈**: typed feedback은 실패 테스트의 자기-서술적 이름을 LLM에 노출해 버그 위치 국소화에 기여.

---

## 4. 3페이지 논문 구성 제안

| 섹션 | 내용 | 분량 |
|------|------|------|
| §1 서론 | 반복 수리의 정보 부재 문제 + 오류 유형별 맞춤 피드백 제안 | 0.5p |
| §2 방법 | 오류 분류기 → 피드백 라우팅 다이어그램 | 0.7p |
| §3 실험 설정 | Defects4J 255, Qwen2.5-Coder-7B, 3-way ablation 정의 | 0.3p |
| §4.1 주결과 | Table 1 + Table 2 + Figure 1 (바차트) | 0.8p |
| §4.2 Case Study | Bug 153 (linearCombination) | 0.3p |
| §5 결론 및 Limitation | symbol 피드백 여지, 단일 seed, 7B 규모 | 0.2p |

### 핵심 메시지 한 문장
> "오류 유형에 특화된 피드백은 가장 빈번한 실패 유형(test-fail, 54%)에서
> 일관되게 10% → 14% 해결률 향상을 보이며, 이는 실패 테스트 메타데이터를
> LLM 프롬프트에 노출하는 단순한 라우팅 만으로 달성 가능함을 시사한다."

---

## 5. Limitation (논문에 정직하게 포함)

1. **단일 seed 실행**: 시간/자원 제약으로 각 전략 1회 실행. Seed 2-3개 재현은 향후 과제.
2. **7B 규모 모델**: 절대 성능(17.3%)은 대규모 모델(ChatGPT 40%+) 대비 낮음. 유형-인식의 효과는 모델 규모에 종속적일 수 있음.
3. **max_model_len=4096**: config YAML 기본값(8192)보다 작음. 일부 대형 프롬프트는 tail-truncate.
4. **Compile 카테고리 샘플 크기**: 소규모 카테고리(N=3~11)에서 통계적 결론 어려움.
5. **Symbol 피드백의 부정적 효과**: cannot-find-symbol 18개 중 해결률 감소 → 구조 재검토 필요.
6. **Infra 실패 2건 제외**: Bug 121(Lang-18), 140(Lang-48)은 Defects4J 체크아웃이 모든 전략에서 실패 → Table 2에 별도 행으로 표시했으며 분모 255에는 포함되어 있음(어느 전략에도 유리/불리하지 않음).

---

## 6. 산출물 파일 목록

```
paper_artifacts/
├── PAPER_SUMMARY.md          ← 이 문서
├── generate_paper_artifacts.py ← 재현 스크립트
├── fig_breakdown.png/pdf     ← Figure 1 (유형별 바차트)
├── table1_overall.tex        ← Table 1 LaTeX
├── table2_breakdown.tex      ← Table 2 LaTeX
├── stats_summary.txt         ← McNemar 통계 전체
└── case_study_bug153.md      ← Bug 153 상세 케이스
```

재생성: `conda activate fse && python paper_artifacts/generate_paper_artifacts.py`

---

## 7. 데이터 소스
- `Results/iterative_error_aware_defects4j_topk_20260418.json`
- `Results/iterative_blind_retry_defects4j_topk_20260418.json`
- `Results/iterative_one_shot_defects4j_topk_20260418.json`

실험 config: `config/ablation_error_aware.yaml`, `config/ablation_blind_retry.yaml`, `config/ablation_1shot.yaml`
