# 2.Bm25.py — BM25 + (옵션) Dense(E5) 2.5단계 리랭크 + 규칙 리랭크 + 캐시
import json
import os
import re
import glob
import argparse
import hashlib
import warnings
from datetime import datetime
from copy import deepcopy
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from tqdm import tqdm

RESULTS_ROOT = Path(os.environ.get("PIPELINE_RESULTS_ROOT", "./Results"))


def _stage_dir(stage: int) -> Path:
    return RESULTS_ROOT / str(stage)

# ---------- 필수: BM25 ----------
try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    raise SystemExit("rank_bm25가 설치되어 있지 않습니다. `pip install rank-bm25` 후 다시 실행하세요.") from e

# 선택 최적화
try:
    import numpy as _np  # type: ignore
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

# 선택: Dense 리랭커 (sentence-transformers 또는 HF AutoModel)
_HAS_ST = False
_torch_mod = None  # 내부 저장 후 아래서 torch에 할당
try:
    import torch as _torch_mod  # type: ignore
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _HAS_ST = True
    except Exception:
        _HAS_ST = False
        from transformers import AutoTokenizer, AutoModel  # type: ignore
except Exception as e:
    print(f"[BM25] torch import 실패: {e}")
    _torch_mod = None
# 외부에서 torch 이름을 안전하게 참조하도록 통일
torch = _torch_mod
_HAS_TORCH = bool(torch)


# ==================== 공통 IO ====================
def _load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: str, obj: Any):
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ==================== 데이터 로드 ====================
def _split_inputs(arg: str) -> List[str]:
    if not arg:
        return []
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    out: List[str] = []
    for part in parts:
        if any(ch in part for ch in ["*", "?", "["]):
            out.extend(sorted(glob.glob(part)))
        else:
            out.append(part)
    return out


def load_all_bugs_from_paths(paths: List[str]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    def _clone_with_internal_meta(obj: Dict[str, Any], internal_key: str, source_path: str) -> Dict[str, Any]:
        cloned = deepcopy(obj) if isinstance(obj, dict) else {}
        cloned["_bm25_internal_key"] = str(internal_key)
        cloned["_bm25_source_path"] = str(source_path)
        return cloned

    for path in paths:
        data = _load_json(path)
        if isinstance(data, dict):
            for raw_k, raw_v in data.items():
                if not isinstance(raw_v, dict):
                    continue
                base_key = str(raw_k)
                key = base_key
                if key in merged:
                    i = 1
                    key = f"{base_key}#{i}"
                    while key in merged:
                        i += 1
                        key = f"{base_key}#{i}"
                merged[key] = _clone_with_internal_meta(raw_v, key, path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                base = str(item.get("bug_id", item.get("id", f"{os.path.basename(path)}:{i}")))
                key = base
                if key in merged:
                    j = 1
                    key = f"{base}#{j}"
                    while key in merged:
                        j += 1
                        key = f"{base}#{j}"
                merged[key] = _clone_with_internal_meta(item, key, path)
        else:
            raise TypeError("all_bugs_meta_data는 dict 또는 list여야 합니다.")
    return merged

def load_real_bugs(path: str) -> Dict[str, Dict[str, Any]]:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise TypeError("real_bugs_by_func는 dict[func_id]->object 형태여야 합니다.")
    return data


def infer_dataset_label_from_path(path: str) -> str:
    name = str(path or "").lower()
    if "bugsinpy" in name:
        return "bugsinpy"
    if "defects4j" in name:
        return "defects4j"
    return ""


def infer_dataset_label_from_entry(entry: Dict[str, Any]) -> str:
    if not isinstance(entry, dict):
        return ""
    for value in (
        entry.get("project_url"),
        entry.get("bug_description_link"),
        entry.get("dataset"),
        entry.get("dataset_name"),
    ):
        text = str(value or "").lower()
        if "bugsinpy" in text:
            return "bugsinpy"
        if "defects4j" in text:
            return "defects4j"
    if entry.get("defects4j_id") is not None:
        return "defects4j"
    if entry.get("bugsinpy_id") is not None:
        return "bugsinpy"
    return ""


def _normalize_dataset_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"defects4j", "bugsinpy"}:
        return text
    return ""


def candidate_dataset_label(
    bug_info: Optional[Dict[str, Any]],
    fallback_dataset: str = "",
    fallback_path: str = "",
) -> str:
    if isinstance(bug_info, dict):
        for key in ("_bm25_dataset", "dataset", "dataset_name"):
            ds = _normalize_dataset_label(bug_info.get(key))
            if ds:
                return ds
        inferred = infer_dataset_label_from_entry(bug_info)
        if inferred:
            return inferred
        source_path = str(bug_info.get("_bm25_source_path") or "")
        if source_path:
            inferred = infer_dataset_label_from_path(source_path)
            if inferred:
                return inferred
    inferred = _normalize_dataset_label(fallback_dataset)
    if inferred:
        return inferred
    return infer_dataset_label_from_path(fallback_path)


def public_candidate_id(bug_info: Dict[str, Any], default_id: str) -> str:
    return str(bug_info.get("bug_id", bug_info.get("id", default_id)))


def candidate_id(
    bug_info: Dict[str, Any],
    default_id: str,
    fallback_dataset: str = "",
    fallback_path: str = "",
) -> str:
    public_id = public_candidate_id(bug_info, default_id)
    if re.match(r"^(defects4j|bugsinpy):", public_id, flags=re.IGNORECASE):
        return public_id
    dataset = candidate_dataset_label(
        bug_info,
        fallback_dataset=fallback_dataset,
        fallback_path=fallback_path,
    )
    if dataset:
        return f"{dataset}:{public_id}"
    return public_id


# ==================== 토크나이즈 ====================
_token_re = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _token_re.findall(text.lower())

def uniq_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ==================== 코퍼스 구성(필드 가중치) ====================
def _repeat_tokens(tokens: List[str], weight: float) -> List[str]:
    if weight <= 0 or not tokens:
        return []
    n = int(weight)
    frac = weight - n
    out = tokens * max(n, 0)
    if frac > 0:
        k = max(1, int(len(tokens) * frac))
        out += tokens[:k]
    return out

def tokens_from_bug_entry(
    entry: Dict[str, Any],
    w_buggy: float = 3.0,
    w_func: float = 0.6,
    w_susp: float = 1.2,
    w_file: float = 0.4,
    w_reason: float = 0.8,
) -> List[str]:
    toks: List[str] = []

    # 1) 핵심 라인
    toks += _repeat_tokens(tokenize(entry.get("buggy_line_content") or ""), w_buggy)

    # 2) 함수 코드 (길이 상한으로 잡음 억제)
    func = entry.get("function") or {}
    func_code = ""
    if isinstance(func, dict):
        func_code = func.get("function_before") or func.get("code") or ""
    if func_code:
        ft = tokenize(func_code)[:256]
        toks += _repeat_tokens(ft, w_func)

    # 3) suspicious nodes (top-5)
    susp = entry.get("suspicious_nodes_topk") or []
    if isinstance(susp, list):
        for node in susp[:5]:
            if not isinstance(node, dict):
                continue
            node_code = node.get("code") or ""
            toks += _repeat_tokens(tokenize(node_code)[:64], w_susp)
            reason = node.get("reason") or ""
            toks += _repeat_tokens(tokenize(reason)[:32], w_reason)

    # 4) 파일/프로젝트 메타
    file_info = entry.get("file") or {}
    if isinstance(file_info, dict):
        toks += _repeat_tokens(tokenize(file_info.get("file_name") or ""), w_file)
        toks += _repeat_tokens(tokenize(file_info.get("file_path") or ""), w_file)

    toks += _repeat_tokens(tokenize(entry.get("project_name") or ""), 0.3)

    return [t for t in toks if t]


def build_bm25_with_cache(
    all_bugs: Dict[str, Dict[str, Any]],
    cache_path: Optional[str],
    bm25_k1: float,
    bm25_b: float,
    **weights,
) -> Tuple[BM25Okapi, List[str], List[List[str]]]:
    # 캐시 fingerprint: 코퍼스 내용 + BM25 파라미터 + 토큰화 규칙 + 필드 가중치
    corpus_snapshot = {
        str(k): all_bugs.get(k, {})
        for k in all_bugs.keys()
    }
    meta_obj = {
        "n": len(all_bugs),
        "doc_keys": [str(k) for k in all_bugs.keys()],
        "corpus_sha1": hashlib.sha1(
            json.dumps(corpus_snapshot, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest(),
        "k1": bm25_k1,
        "b": bm25_b,
        "weights": weights,
        "token_re": _token_re.pattern,
    }
    corpus_fp = hashlib.sha1(
        json.dumps(meta_obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    if cache_path and os.path.exists(cache_path):
        try:
            cache = _load_json(cache_path)
            if cache.get("meta", {}).get("fingerprint") == corpus_fp:
                tokenized_corpus = cache["tokenized_corpus"]
                bug_id_list = cache.get("doc_key_list") or cache["bug_id_list"]
                bm25 = BM25Okapi(tokenized_corpus, k1=bm25_k1, b=bm25_b)
                return bm25, bug_id_list, tokenized_corpus
        except Exception:
            pass

    bug_id_list: List[str] = []
    tokenized_corpus: List[List[str]] = []

    for k, v in all_bugs.items():
        doc_key = str(k)
        tokens = tokens_from_bug_entry(v, **weights)
        bug_id_list.append(doc_key)
        tokenized_corpus.append(tokens)

    if not tokenized_corpus or all(len(t) == 0 for t in tokenized_corpus):
        raise ValueError("BM25 코퍼스를 만들 수 없습니다(토큰 비어 있음).")

    bm25 = BM25Okapi(tokenized_corpus, k1=bm25_k1, b=bm25_b)

    # 캐시 저장
    if cache_path:
        try:
            _save_json(cache_path, {
                "meta": {
                    "fingerprint": corpus_fp,
                    "n": len(all_bugs),
                    "k1": bm25_k1,
                    "b": bm25_b,
                    "weights": weights,
                    "token_re": _token_re.pattern,
                },
                "doc_key_list": bug_id_list,
                "bug_id_list": bug_id_list,
                "tokenized_corpus": tokenized_corpus,
            })
        except Exception:
            pass

    return bm25, bug_id_list, tokenized_corpus


# ==================== 쿼리 선택/확장 ====================
def pick_query_from_func_obj(func_obj: Dict[str, Any]) -> Tuple[str, str]:
    bug_line = func_obj.get("bug_line", {}) if isinstance(func_obj, dict) else {}
    candidates = [
        ("buggy_line_content", func_obj.get("buggy_line_content")),
        ("bug_line.code", (bug_line.get("code") if isinstance(bug_line, dict) else None)),
        ("code", func_obj.get("code")),
        ("code_line", func_obj.get("code_line")),
    ]
    s_list = func_obj.get("suspicious_nodes_topk")
    if isinstance(s_list, list) and s_list:
        first = s_list[0]
        if isinstance(first, dict):
            candidates.append(("suspicious_nodes_topk[0].code", first.get("code")))
    for name, c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip(), name

    # 폴백: 상위 3개 suspicious code 이어붙이기
    if isinstance(s_list, list) and s_list:
        codes: List[str] = []
        for node in s_list[:3]:
            if isinstance(node, dict):
                code = node.get("code")
                if isinstance(code, str) and code.strip():
                    codes.append(code.strip())
        joined = " ".join(codes).strip()
        if joined:
            return joined, "suspicious_nodes_topk[:3].code_joined"
    return "", "EMPTY"


_REASON_EXPAND = [
    ("indexerror", ["index", "out", "range", "len", "range", "bounds"]),
    ("out of range", ["index", "range", "bounds", "len"]),
    ("off-by-one", ["index", "range", "len"]),
    ("keyerror", ["dict", "get", "in", "keys"]),
    ("typeerror", ["isinstance", "type", "cast"]),
    ("valueerror", ["validate", "check", "empty"]),
    ("zerodivision", ["divide", "zero", "float"]),
    ("none", ["is", "==", "null", "empty"]),
    ("empty", ["len", "==", "0"]),
    ("boundary", ["min", "max", "clip"]),
]
_CODE_CUES_EXPAND = [
    ("range(", ["index", "len"]),
    ("len(", ["index", "empty"]),
    ("[", ["index"]),
    (".get(", ["dict", "key"]),
    ("try:", ["except"]),
    ("except", ["try"]),
    ("is None", ["none"]),
    ("== None", ["none"]),
]

def expand_query_tokens(tokens: List[str], func_obj: Dict[str, Any]) -> List[str]:
    out = list(tokens)

    # reason 기반 확장
    susp = func_obj.get("suspicious_nodes_topk") or []
    reasons = []
    if isinstance(susp, list):
        for node in susp[:5]:
            if isinstance(node, dict) and isinstance(node.get("reason"), str):
                reasons.append(node["reason"].lower())
    reason_text = " ".join(reasons)
    for k, adds in _REASON_EXPAND:
        if k in reason_text:
            out += adds

    # 코드 패턴 기반 확장(쿼리/buggy_line 텍스트)
    bl = func_obj.get("buggy_line_content") or ""
    base_text = " ".join(tokens)
    for cue, adds in _CODE_CUES_EXPAND:
        if cue in bl or cue in base_text:
            out += adds
    return uniq_keep_order(out)


# ==================== 유틸 ====================
def get_current_ids(
    func_id: Any,
    func_obj: Optional[Dict[str, Any]],
    fallback_dataset: str = "",
    fallback_path: str = "",
) -> set:
    s = {str(func_id)}
    if isinstance(func_obj, dict):
        s.add(candidate_id(func_obj, str(func_id), fallback_dataset=fallback_dataset, fallback_path=fallback_path))
        for key in ("id", "bug_id"):
            v = func_obj.get(key)
            if v is not None:
                s.add(str(v))
    return s

def top_indices_by_score(scores: List[float], top_n: int) -> List[int]:
    n = len(scores)
    if top_n >= n:
        return sorted(range(n), key=lambda i: scores[i], reverse=True)
    if _HAS_NUMPY:
        scores_np = _np.array(scores)
        idx = _np.argpartition(-scores_np, top_n - 1)[:top_n]
        return idx[_np.argsort(-scores_np[idx])].tolist()
    else:
        return sorted(range(n), key=lambda i: scores[i], reverse=True)[:top_n]

def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mx = max(scores)
    mn = min(scores)
    if mx == mn:
        return [0.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


# ==================== 규칙 리랭커 ====================
def _file_ext(name_or_path: Optional[str]) -> str:
    if not name_or_path:
        return ""
    base = os.path.basename(name_or_path)
    _, ext = os.path.splitext(base)
    return ext.lower()

def _token_jaccard(a: str, b: str) -> float:
    sa = set(tokenize(a))
    sb = set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def rule_rerank_candidates(
    base_list: List[Tuple[str, Dict[str, Any], float]],
    query_obj: Dict[str, Any],
    bonus_same_project: float = 0.10,
    bonus_same_ext: float = 0.05,
    bonus_token_jaccard: float = 0.5,
) -> List[Tuple[str, Dict[str, Any], float]]:
    q_proj = (query_obj.get("project_name") or "").lower()
    q_file = ""
    fi = query_obj.get("file") or {}
    if isinstance(fi, dict):
        q_file = fi.get("file_name") or fi.get("file_path") or ""
    q_ext = _file_ext(q_file)
    q_line = query_obj.get("buggy_line_content") or ""

    reranked = []
    for bug_key, bug_info, base_s in base_list:
        s = base_s
        proj = (bug_info.get("project_name") or "").lower()
        if q_proj and proj and proj == q_proj:
            s += bonus_same_project
        file_info = bug_info.get("file") or {}
        cand_file = ""
        if isinstance(file_info, dict):
            cand_file = file_info.get("file_name") or file_info.get("file_path") or ""
        if q_ext and _file_ext(cand_file) == q_ext:
            s += bonus_same_ext
        cand_line = bug_info.get("buggy_line_content") or ""
        jac = _token_jaccard(q_line, cand_line)
        s += bonus_token_jaccard * jac
        reranked.append((bug_key, bug_info, s))
    reranked.sort(key=lambda x: x[2], reverse=True)
    return reranked


# ==================== Dense 리랭커(E5) ====================
def _device_from_str(s: str) -> str:
    if s == "auto":
        if _HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return s

def _l2_normalize(x: _np.ndarray, axis: int = 1) -> _np.ndarray:
    denom = _np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12
    return x / denom

def _mean_pool(last_hidden_state: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    s = (last_hidden_state * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1e-9)
    return s / d

if not _HAS_TORCH:
    class DenseEncoder:
        """Placeholder when torch is unavailable."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Dense 리랭커를 쓰려면 torch가 필요합니다.")
else:
    class DenseEncoder:
        """
        E5 계열 임베딩. sentence-transformers가 있으면 우선 사용.
        없으면 HF AutoModel + mean pooling.
        """
        def __init__(self, model_name: str, device: str = "auto", batch_size: int = 64):
            if torch is None:
                raise RuntimeError("torch가 없습니다.")
            self.model_name = model_name
            self.device = _device_from_str(device)
            self.batch_size = max(1, int(batch_size))
            self.model = None
            self.tokenizer = None
            self._use_st = False

            if _HAS_ST:
                try:
                    self.model = SentenceTransformer(model_name, device=self.device)
                    self._use_st = True
                except Exception:
                    warnings.warn("sentence-transformers 로드 실패, HF AutoModel로 폴백합니다.")
                    self._use_st = False

            if not self._use_st:
                try:
                    from transformers import AutoTokenizer, AutoModel  # type: ignore
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModel.from_pretrained(model_name).to(self.device)
                    self.model.eval()
                except Exception as e:
                    raise RuntimeError(f"Dense 모델 로드 실패: {e}")

        def encode(self, texts: List[str], is_query: bool = False) -> _np.ndarray:
            if torch is None:
                raise RuntimeError("torch가 없습니다.")
            if not texts:
                return _np.zeros((0, 384), dtype=_np.float32)  # safe default
            # E5 권장 prefix
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + (t or "") for t in texts]

            if self._use_st:
                v = self.model.encode(
                    texts, batch_size=self.batch_size, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False
                )
                return v.astype(_np.float32, copy=False)

            # HF AutoModel 경로 (mean pooling + L2 norm)
            embs = []
            # Inference-only: avoid autograd graph buildup on GPU
            with torch.inference_mode():
                for i in range(0, len(texts), self.batch_size):
                    chunk = texts[i:i + self.batch_size]
                    toks = self.tokenizer(
                        chunk, padding=True, truncation=True, max_length=512, return_tensors="pt"
                    ).to(self.device)
                    out = self.model(**toks)
                    pooled = _mean_pool(out.last_hidden_state, toks["attention_mask"])
                    if hasattr(pooled, "detach"):
                        pooled = pooled.detach()
                    if hasattr(pooled, "cpu"):
                        pooled = pooled.cpu()
                    embs.append(pooled)
            x = torch.cat(embs, dim=0)
            if getattr(x, "requires_grad", False):
                x = x.detach()
            if hasattr(x, "cpu"):
                x = x.cpu()
            x = x.numpy().astype(_np.float32, copy=False)
            return _l2_normalize(x, axis=1)

def _build_doc_text(entry: Dict[str, Any]) -> str:
    """Dense 인덱스에 넣을 문서 텍스트 구성(가벼운 요약)."""
    parts = []
    parts.append(entry.get("buggy_line_content") or "")
    file_info = entry.get("file") or {}
    if isinstance(file_info, dict):
        parts.append(str(file_info.get("file_name") or ""))
        parts.append(str(file_info.get("file_path") or ""))
    susp = entry.get("suspicious_nodes_topk") or []
    reasons = []
    if isinstance(susp, list):
        for node in susp[:3]:
            if isinstance(node, dict):
                r = node.get("reason")
                if isinstance(r, str) and r.strip():
                    reasons.append(r.strip())
    if reasons:
        parts.append(" | ".join(reasons))
    # 함수 본문은 너무 길어 잡음이니 상위 120 토큰만
    func = entry.get("function") or {}
    func_code = ""
    if isinstance(func, dict):
        func_code = func.get("function_before") or func.get("code") or ""
    if func_code:
        t = tokenize(func_code)[:120]
        parts.append(" ".join(t))
    return " \n ".join([p for p in parts if p])

def _build_query_text(func_obj: Dict[str, Any]) -> str:
    parts = []
    parts.append(func_obj.get("buggy_line_content") or "")
    file_info = func_obj.get("file") or {}
    if isinstance(file_info, dict):
        parts.append(str(file_info.get("file_name") or ""))
        parts.append(str(file_info.get("file_path") or ""))
    susp = func_obj.get("suspicious_nodes_topk") or []
    reasons = []
    if isinstance(susp, list):
        for node in susp[:5]:
            if isinstance(node, dict):
                r = node.get("reason")
                if isinstance(r, str) and r.strip():
                    reasons.append(r.strip())
    if reasons:
        parts.append(" | ".join(reasons))
    return " \n ".join([p for p in parts if p])

def _load_or_build_dense_corpus(
    all_bugs: Dict[str, Dict[str, Any]],
    encoder: DenseEncoder,
    bug_id_list: List[str],
    dense_cache: str,
) -> Tuple[_np.ndarray, Dict[str, int]]:
    """
    전체 코퍼스 임베딩 캐시(.npz) 로드 또는 생성.
    반환: (embeddings[N, D], id2row)
    """
    os.makedirs(os.path.dirname(dense_cache) or ".", exist_ok=True)
    if os.path.exists(dense_cache):
        try:
            data = _np.load(dense_cache, allow_pickle=True)
            ids = data["ids"].tolist()
            embs = data["embeddings"].astype(_np.float32)
            meta = json.loads(data["meta"].item())
            if meta.get("model") == encoder.model_name and ids == bug_id_list:
                id2row = {str(i): idx for idx, i in enumerate(ids)}
                return embs, id2row
        except Exception:
            warnings.warn("Dense 캐시 로드 실패. 새로 생성합니다.")

    # 새로 생성
    docs = []
    for k in bug_id_list:
        entry = all_bugs.get(k, {})
        docs.append(_build_doc_text(entry))
    embs = encoder.encode(docs, is_query=False).astype(_np.float32, copy=False)
    ids_arr = _np.array(bug_id_list, dtype=object)
    meta = _np.array(json.dumps({"model": encoder.model_name}), dtype=object)
    _np.savez_compressed(dense_cache, embeddings=embs, ids=ids_arr, meta=meta)
    id2row = {str(i): idx for idx, i in enumerate(bug_id_list)}
    return embs, id2row

def dense_rerank_over_m(
    base_idxs: List[int],
    bug_id_list: List[str],
    bm25_norm_scores: List[float],
    all_bugs: Dict[str, Dict[str, Any]],
    encoder: DenseEncoder,
    dense_corpus: _np.ndarray,
    id2row: Dict[str, int],
    func_obj: Dict[str, Any],
    dense_weight: float = 0.6,
) -> List[Tuple[str, Dict[str, Any], float]]:
    """
    BM25 상위 M 후보들에 대해 Dense 점수와 결합.
    결합: final = (1-dense_weight)*bm25_norm + dense_weight*sim_norm
    """
    # 쿼리 임베딩
    q_text = _build_query_text(func_obj)
    q_emb = encoder.encode([q_text], is_query=True)  # (1, D)
    q = q_emb[0:1]  # (1,D)

    # 후보 코사인 유사도 (임베딩 L2 정규화되어 있음 → dot == cosine)
    c_rows = []
    for i in base_idxs:
        bid = bug_id_list[i]
        row = id2row.get(bid, None)
        if row is None:
            # 캐시에 없으면 on-the-fly 임베딩
            cand_text = _build_doc_text(all_bugs.get(bid, {}))
            v = encoder.encode([cand_text], is_query=False)[0]
            if v.ndim == 1:
                v = v.reshape(1, -1)
            sim = float((_np.dot(q, v.T))[0][0])
            c_rows.append(sim)
            continue
        v = dense_corpus[row:row+1]  # (1,D)
        sim = float((_np.dot(q, v.T))[0][0])  # cosine
        c_rows.append(sim)

    # Dense sim 정규화 (per-query min-max → 0..1)
    sim_norm = _normalize_scores(c_rows)

    # 결합 점수
    out: List[Tuple[str, Dict[str, Any], float]] = []
    for rank_idx, idx in enumerate(base_idxs):
        bid = bug_id_list[idx]
        info = all_bugs.get(bid, {})
        bm = bm25_norm_scores[rank_idx]
        ds = sim_norm[rank_idx]
        final = (1.0 - dense_weight) * bm + dense_weight * ds
        out.append((bid, info, final))
    # 결합 점수 기준 정렬
    out.sort(key=lambda x: x[2], reverse=True)
    return out


# ==================== 메인 ====================
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run_tag", default="full", help="Run tag used for auto I/O path discovery.")
    parser.add_argument("--all_bugs", default=None,
                        help="코퍼스(all bugs) JSON 경로(또는 glob/CSV)")
    parser.add_argument("--input_json", "--in_json", dest="input_json",
                        default=None,
                        help="실제 버그(by func) JSON 경로(또는 glob/CSV)")
    parser.add_argument("--output_json", "--out_json", dest="output_json", default=None,
                        help="출력 파일(JSON) - 입력이 여러 개면 자동 파생")

    # ---- 권장 튜닝 기본값 ----
    parser.add_argument("--topk", type=int, default=10, help="최종 상위 후보 개수")
    parser.add_argument("--stage_topm", type=int, default=200, help="1단계 BM25 상위 M(리랭크 전에)")
    parser.add_argument("--bm25_k1", type=float, default=1.6, help="BM25 k1")
    parser.add_argument("--bm25_b", type=float, default=0.7, help="BM25 b")
    parser.add_argument("--corpus_cache", default=str(_stage_dir(2) / "cache/bm25_corpus_cache.json"),
                        help="BM25 코퍼스 토큰 캐시(JSON)")

    # 필드 가중치
    parser.add_argument("--w_buggy", type=float, default=3.0, help="buggy_line_content 가중치")
    parser.add_argument("--w_func", type=float, default=0.6, help="function_before/code 가중치")
    parser.add_argument("--w_susp", type=float, default=1.2, help="suspicious_nodes_topk.code 가중치")
    parser.add_argument("--w_reason", type=float, default=0.8, help="suspicious_nodes_topk.reason 가중치")
    parser.add_argument("--w_file", type=float, default=0.4, help="파일명/경로 가중치")

    # 쿼리 확장/규칙 리랭크
    parser.add_argument("--no_expand", action="store_true", help="쿼리 확장 비활성화")
    parser.add_argument("--bonus_same_project", type=float, default=0.10, help="동일 프로젝트 보너스")
    parser.add_argument("--bonus_same_ext", type=float, default=0.05, help="동일 확장자 보너스")
    parser.add_argument("--bonus_token_jaccard", type=float, default=0.5, help="버그라인 토큰 자카드 가중치")

    # ---- Dense 리랭커 옵션 (기본: 비활성화) ----
    parser.add_argument("--dense_enable", action="store_true", default=False, help="Enable Dense(E5) rerank")
    parser.add_argument("--dense_model", default="/home/selab/models/e5-small-v2", help="E5 계열 임베딩 모델 이름 또는 로컬 경로")
    parser.add_argument("--dense_device", default="auto", choices=["auto", "cpu", "cuda"], help="임베딩 장치")
    parser.add_argument("--dense_batch", type=int, default=1, help="임베딩 배치 사이즈")
    parser.add_argument("--dense_cache", default=str(_stage_dir(2) / "cache/dense_e5_small_v2.npz"),
                        help="Dense 코퍼스 임베딩 캐시(.npz)")
    parser.add_argument("--dense_weight", type=float, default=0.6,
                        help="결합 가중치: final=(1-w)*bm25_norm + w*dense_norm")
    parser.add_argument(
        "--single_only",
        dest="single_only",
        action="store_true",
        help="Force single-only discovery (never prefer .hard inputs).",
    )
    parser.add_argument(
        "--no_single_only",
        dest="single_only",
        action="store_false",
        help="Disable single-only discovery and allow hard-priority auto input selection.",
    )
    parser.add_argument(
        "--same_dataset_only",
        dest="same_dataset_only",
        action="store_true",
        help="Restrict BM25 corpus to the same dataset as the real_bugs input.",
    )
    parser.add_argument(
        "--no_same_dataset_only",
        dest="same_dataset_only",
        action="store_false",
        help="Allow cross-dataset BM25 corpus mixing.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print resolved config and exit.")
    parser.set_defaults(single_only=True, same_dataset_only=False)

    args = parser.parse_args()

    def _auto_output_path(run_tag: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(_stage_dir(2) / f"2.{run_tag}.{ts}.json")

    def _extract_ts(name: str) -> str:
        m = re.search(r"(\d{8}_\d{6})", name)
        return m.group(1) if m else ""

    def _discover_root_metadata_inputs() -> List[str]:
        repo_root = Path(__file__).resolve().parent
        candidates = [
            repo_root / "bugsinpy_bugs_meta_data.json",
            repo_root / "defects4j_bugs_meta_data.json",
        ]
        return [str(p) for p in candidates if p.is_file()]

    def _iter_entry_samples(data: Any, limit: int = 8):
        if isinstance(data, dict):
            count = 0
            for value in data.values():
                if isinstance(value, dict):
                    yield value
                    count += 1
                    if count >= limit:
                        break
        elif isinstance(data, list):
            count = 0
            for value in data:
                if isinstance(value, dict):
                    yield value
                    count += 1
                    if count >= limit:
                        break

    def _looks_ast_enriched_json_path(path: Path) -> bool:
        name = path.name.lower()
        if "ast_analyzed" in name:
            return True
        try:
            data = _load_json(str(path))
        except Exception:
            return False
        for entry in _iter_entry_samples(data):
            if "suspicious_nodes_topk" in entry:
                return True
        return False

    def _prefer_ast_enriched_paths(files: List[Path], reason_base: str) -> Tuple[List[Path], str]:
        ast_named = [p for p in files if "ast_analyzed" in p.name.lower()]
        if ast_named:
            return ast_named, reason_base + ":ast_named"
        ast_content = [p for p in files if _looks_ast_enriched_json_path(p)]
        if ast_content:
            return ast_content, reason_base + ":ast_content"
        return files, reason_base

    def _find_latest_manifest(run_tag: str) -> Optional[str]:
        stage_dir = RESULTS_ROOT / "1"
        if not stage_dir.is_dir():
            return None
        def _pick(pattern: str) -> Optional[str]:
            files = [p for p in stage_dir.glob(pattern) if p.is_file()]
            if not files:
                return None
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(files[0])

        if run_tag:
            tagged = _pick(f"1.{run_tag}*.manifest.json")
            if tagged:
                return tagged
        by_prefix = _pick("1.*.manifest.json")
        return by_prefix

    def _discover_stage1_hard_inputs(run_tag: str, single_only: bool = False) -> Tuple[List[str], Dict[str, str]]:
        manifest_path = _find_latest_manifest(run_tag)
        if manifest_path:
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    man = json.load(f)
                splits = man.get("splits") if isinstance(man, dict) else {}
                out: List[str] = []
                reason: Dict[str, str] = {}
                keys = (
                    ("defects4j.single", "bugsinpy.single")
                    if bool(single_only)
                    else ("defects4j.hard", "bugsinpy.hard")
                )
                for k in keys:
                    p = splits.get(k) if isinstance(splits, dict) else None
                    if isinstance(p, str) and Path(p).is_file():
                        out.append(p)
                        reason[p] = f"manifest:{k}"
                if out:
                    return out, reason
            except Exception:
                pass

        discovered_inputs, discovered_reasons = _discover_stage_inputs(
            prev_stage=1,
            run_tag=run_tag,
            prefer_hard=(not bool(single_only)),
            single_only=bool(single_only),
        )
        if discovered_inputs:
            return discovered_inputs, discovered_reasons

        root_inputs = _discover_root_metadata_inputs()
        if root_inputs:
            return root_inputs, {p: "root_metadata:fallback_only" for p in root_inputs}

        return [], {}

    def _discover_stage_inputs(
        prev_stage: int,
        run_tag: str,
        prefer_hard: bool = False,
        single_only: bool = False,
    ) -> Tuple[List[str], Dict[str, str]]:
        files: List[Path] = []
        stage_dir = _stage_dir(prev_stage)
        if stage_dir.is_dir():
            files.extend([
                p for p in stage_dir.glob("*.json")
                if p.is_file() and not p.name.endswith(".manifest.json")
            ])
        if not files:
            return [], {}
        reason_base = "scan:all"
        requested_tag = (run_tag or "").strip().lower()
        if requested_tag:
            tagged = [p for p in files if requested_tag in p.name.lower()]
            if tagged:
                files = tagged
                reason_base = "scan:run_tag"
            elif requested_tag == "full":
                non_smoke = [p for p in files if "smoke" not in p.name.lower()]
                if non_smoke:
                    files = non_smoke
                    reason_base = "scan:full_non_smoke"
        if bool(single_only):
            prefer_hard = False
            single_files = [p for p in files if ".single" in p.name]
            if single_files:
                non_single_non_hard = [p for p in files if ".single" not in p.name and ".hard" not in p.name]
                hard_files = [p for p in files if ".hard" in p.name]
                files = single_files + non_single_non_hard + hard_files
                reason_base = "scan:single_priority"
            else:
                non_hard = [p for p in files if ".hard" not in p.name]
                hard_files = [p for p in files if ".hard" in p.name]
                if non_hard:
                    files = non_hard + hard_files
                    reason_base = "scan:non_hard_priority"
                elif hard_files:
                    files = hard_files
                    reason_base = "scan:hard_fallback"
        elif prefer_hard:
            hard = [p for p in files if ".hard" in p.name]
            if hard:
                files = hard
                reason_base = "scan:hard_priority"
        if not files:
            return [], {}
        if prev_stage in (1, 2):
            files, reason_base = _prefer_ast_enriched_paths(files, reason_base)
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        ts_list = [_extract_ts(p.name) for p in files if _extract_ts(p.name)]
        ts_reason = ""
        if ts_list:
            latest_ts = max(ts_list)
            no_ts = [p for p in files if not _extract_ts(p.name)]
            grouped = [p for p in files if latest_ts in p.name]
            if grouped:
                files = grouped + no_ts
                ts_reason = f":latest_ts:{latest_ts}"
        # Keep newest first, deduplicate same path string
        out: List[str] = []
        seen = set()
        seen_ds = set()
        reason: Dict[str, str] = {}
        for p in files:
            s = str(p)
            if s in seen:
                continue
            name = p.name.lower()
            ds = "defects4j" if "defects4j" in name else ("bugsinpy" if "bugsinpy" in name else "")
            if ds and ds in seen_ds:
                continue
            seen.add(s)
            if ds:
                seen_ds.add(ds)
            out.append(s)
            reason[s] = reason_base + ts_reason
        return out, reason

    def _discover_latest_stage_json(
        prev_stage: int,
        run_tag: str,
        single_only: bool = False,
    ) -> Tuple[Optional[str], str]:
        found, reasons = _discover_stage_inputs(
            prev_stage=prev_stage,
            run_tag=run_tag,
            prefer_hard=False,
            single_only=bool(single_only),
        )
        if found:
            return found[0], reasons.get(found[0], "discover:first")
        files: List[Path] = []
        stage_dir = _stage_dir(prev_stage)
        if stage_dir.is_dir():
            files.extend([
                p for p in stage_dir.glob("*.json")
                if p.is_file() and not p.name.endswith(".manifest.json")
            ])
        if not files:
            return None, ""
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(files[0]), "fallback:latest_json"

    def _results_subdir_for(input_path: str, subdir: str, default_out: str) -> str:
        parts = os.path.normpath(input_path).split(os.sep)
        if RESULTS_ROOT.name in parts:
            idx = parts.index(RESULTS_ROOT.name)
            return os.path.join(*parts[:idx + 1], subdir)
        return os.path.dirname(default_out) or str(_stage_dir(int(subdir)))

    def _derive_output_path(input_path: str, default_out: str) -> str:
        name = os.path.basename(input_path)
        ds_label = infer_dataset_label_from_path(input_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if name.startswith("2.LLM_Analyzed_Results"):
            name = name.replace("2.LLM_Analyzed_Results", "2.BM25Result", 1)
        elif name.startswith("2.AST_Analyzed_Results"):
            name = name.replace("2.AST_Analyzed_Results", "2.BM25Result", 1)
        elif name.startswith("1.AST_Analyzed_Results"):
            name = name.replace("1.AST_Analyzed_Results", "2.BM25Result", 1)
        else:
            name = f"2.BM25Result.{ds_label or 'unknown'}.{args.run_tag}.{ts}.json"
        # 데이터셋 라벨이 출력 파일명에 없으면 삽입
        if ds_label and ds_label not in name.lower():
            stem, ext = os.path.splitext(name)
            name = f"{stem}.{ds_label}{ext}"
        out_dir = _results_subdir_for(input_path, "2", default_out)
        return os.path.join(out_dir, name)

    # 데이터 로드
    auto_all = ""
    all_input_reasons: Dict[str, str] = {}
    if args.all_bugs:
        all_bug_inputs = _split_inputs(args.all_bugs)
        if not all_bug_inputs:
            all_bug_inputs = [args.all_bugs]
        all_input_reasons = {p: "explicit:all_bugs_arg" for p in all_bug_inputs}
    else:
        auto_all_inputs, auto_all_reasons = _discover_stage1_hard_inputs(
            str(args.run_tag),
            single_only=bool(args.single_only),
        )
        if not auto_all_inputs:
            raise SystemExit(
                "No Stage-1 AST-enriched JSON found for auto discovery in Results/1, "
                "and no root metadata fallback JSON found "
                "(bugsinpy_bugs_meta_data.json / defects4j_bugs_meta_data.json)."
            )
        all_bug_inputs = auto_all_inputs
        all_input_reasons = dict(auto_all_reasons)
        auto_all = ",".join(auto_all_inputs)
        print(f"[auto] all_bugs <- {auto_all}")
        for p in auto_all_inputs:
            print(f"[auto][all_bugs] selected_input_file={p} selection_reason={all_input_reasons.get(p, '')}")
    auto_real = ""
    real_input_reasons: Dict[str, str] = {}
    if args.input_json:
        real_bug_inputs = _split_inputs(args.input_json)
        if not real_bug_inputs:
            real_bug_inputs = [args.input_json]
        real_input_reasons = {p: "explicit:real_bugs_arg" for p in real_bug_inputs}
    else:
        auto_real_inputs, auto_real_reasons = _discover_stage_inputs(
            prev_stage=1,
            run_tag=str(args.run_tag),
            prefer_hard=(not bool(args.single_only)),
            single_only=bool(args.single_only),
        )
        if not auto_real_inputs:
            auto_one, auto_one_reason = _discover_latest_stage_json(
                1,
                str(args.run_tag),
                single_only=bool(args.single_only),
            )
            auto_real_inputs = [auto_one] if auto_one else []
            auto_real_reasons = {auto_one: auto_one_reason} if auto_one else {}
        if not auto_real_inputs:
            raise SystemExit(f"No Stage-1 JSON found for auto discovery in {RESULTS_ROOT / '1'}.")
        real_bug_inputs = auto_real_inputs
        real_input_reasons = dict(auto_real_reasons)
        auto_real = ",".join(auto_real_inputs)
        print(f"[auto] in_json <- {auto_real}")
        for p in auto_real_inputs:
            print(f"[auto][in_json] selected_input_file={p} selection_reason={real_input_reasons.get(p, '')}")

    default_out = args.output_json or _auto_output_path(str(args.run_tag))
    if args.output_json and len(real_bug_inputs) == 1:
        resolved_outputs = [str(args.output_json)]
    else:
        resolved_outputs = [
            _derive_output_path(real_path, default_out)
            for real_path in real_bug_inputs
        ]

    print(
        f"[resolved][Stage2] run_tag={args.run_tag} topk={int(args.topk)} "
        f"single_only={bool(args.single_only)} same_dataset_only={bool(args.same_dataset_only)} "
        f"stage_topm={int(args.stage_topm)} dry_run={bool(args.dry_run)}"
    )
    print(f"[resolved][Stage2] all_bugs={','.join(all_bug_inputs)}")
    print(f"[resolved][Stage2] in_json={','.join(real_bug_inputs)}")
    for i, out_p in enumerate(resolved_outputs):
        print(f"[resolved][Stage2] out_json[{i}]={out_p}")
    if args.dry_run:
        print("[dry_run] Stage2 exiting before BM25 retrieval.")
        return

    corpus_bundle_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    dense_encoder = None
    if args.dense_enable:
        if not _HAS_TORCH:
            warnings.warn("torch가 없어 Dense 리랭크를 비활성화합니다.")
        elif args.dense_model and not os.path.exists(args.dense_model):
            warnings.warn(f"Dense 모델 경로가 없어 비활성화합니다: {args.dense_model}")
        else:
            try:
                dense_encoder = DenseEncoder(args.dense_model, device=args.dense_device, batch_size=args.dense_batch)
            except Exception as e:
                warnings.warn(f"Dense 리랭커 초기화 실패: {e}\nBM25+규칙 리랭크만 사용합니다.")

    for real_path, out_path in zip(real_bug_inputs, resolved_outputs):
        selected_all_bug_inputs = list(all_bug_inputs)
        target_dataset = infer_dataset_label_from_path(real_path)
        if bool(args.same_dataset_only) and target_dataset:
            filtered_inputs = [p for p in all_bug_inputs if infer_dataset_label_from_path(p) == target_dataset]
            if filtered_inputs:
                selected_all_bug_inputs = filtered_inputs
        corpus_key = tuple(sorted(selected_all_bug_inputs))
        bundle = corpus_bundle_cache.get(corpus_key)
        if bundle is None:
            selected_all_bugs = load_all_bugs_from_paths(selected_all_bug_inputs)
            if bool(args.same_dataset_only) and target_dataset:
                selected_all_bugs = {
                    k: v
                    for k, v in selected_all_bugs.items()
                    if infer_dataset_label_from_entry(v) in {"", target_dataset}
                }
            corpus_tag = target_dataset or "mixed"
            corpus_cache_path = str(args.corpus_cache)
            if corpus_cache_path:
                stem, ext = os.path.splitext(corpus_cache_path)
                ext = ext or ".json"
                corpus_cache_path = f"{stem}.{corpus_tag}{ext}"
            bm25, bug_id_list, _ = build_bm25_with_cache(
                selected_all_bugs,
                cache_path=corpus_cache_path,
                bm25_k1=args.bm25_k1,
                bm25_b=args.bm25_b,
                w_buggy=args.w_buggy,
                w_func=args.w_func,
                w_susp=args.w_susp,
                w_file=args.w_file,
                w_reason=args.w_reason,
            )
            dense_corpus = None
            id2row = None
            dense_ready = False
            dense_cache_path = str(args.dense_cache)
            if dense_encoder is not None:
                if dense_cache_path:
                    stem, ext = os.path.splitext(dense_cache_path)
                    ext = ext or ".npz"
                    dense_cache_path = f"{stem}.{corpus_tag}{ext}"
                try:
                    dense_corpus, id2row = _load_or_build_dense_corpus(
                        selected_all_bugs,
                        dense_encoder,
                        bug_id_list,
                        dense_cache_path,
                    )
                    dense_ready = True
                except Exception as e:
                    warnings.warn(f"Dense 코퍼스 준비 실패({corpus_tag}): {e}\nBM25+규칙 리랭크만 사용합니다.")
            bundle = {
                "all_bugs": selected_all_bugs,
                "bug_id_list": bug_id_list,
                "bm25": bm25,
                "dense_ready": dense_ready,
                "dense_corpus": dense_corpus,
                "id2row": id2row,
                "dense_cache_path": dense_cache_path,
            }
            corpus_bundle_cache[corpus_key] = bundle

        all_bugs = bundle["all_bugs"]
        bug_id_list = bundle["bug_id_list"]
        bm25 = bundle["bm25"]
        dense_ready = bool(bundle["dense_ready"])
        dense_corpus = bundle["dense_corpus"]
        id2row = bundle["id2row"]
        real_bugs_by_func = load_real_bugs(real_path)
        final_results: Dict[str, Any] = {}
        it = tqdm(list(real_bugs_by_func.items()), desc="BM25 + Dense ReRank", disable=False)

        for func_id, func_obj in it:
            out_obj = deepcopy(func_obj if isinstance(func_obj, dict) else {})

            query_text, source = pick_query_from_func_obj(func_obj if isinstance(func_obj, dict) else {})
            base_tokens = tokenize(query_text)
            if not args.no_expand:
                base_tokens = expand_query_tokens(base_tokens, func_obj if isinstance(func_obj, dict) else {})
            tokenized_query = base_tokens

            top_jsons: List[Dict[str, Any]] = []
            if tokenized_query:
                raw_scores = bm25.get_scores(tokenized_query).tolist()

                # 1단계: BM25 상위 M 추출
                M = max(args.topk, args.stage_topm)
                idxs = top_indices_by_score(raw_scores, M)

                # 0~1 정규화 후 베이스 리스트 구성
                raw_sel = [raw_scores[i] for i in idxs]
                raw_norm = _normalize_scores(raw_sel)

                base_list: List[Tuple[str, Dict[str, Any], float]] = []
                for rank_idx, idx in enumerate(idxs):
                    bug_key = bug_id_list[idx]
                    bug_info = all_bugs.get(bug_key, {})
                    base_list.append((bug_key, bug_info, raw_norm[rank_idx]))

                # 자기 자신 제외 셋
                current_ids = get_current_ids(
                    func_id,
                    func_obj if isinstance(func_obj, dict) else None,
                    fallback_dataset=target_dataset,
                    fallback_path=real_path,
                )

                # 2.5단계: Dense 결합 리랭크 (가능하면)
                if dense_ready and dense_encoder is not None and dense_corpus is not None and id2row is not None:
                    # BM25 top-M의 인덱스로 dense 결합
                    dense_combined = dense_rerank_over_m(
                        base_idxs=idxs,
                        bug_id_list=bug_id_list,
                        bm25_norm_scores=raw_norm,
                        all_bugs=all_bugs,
                        encoder=dense_encoder,
                        dense_corpus=dense_corpus,
                        id2row=id2row,
                        func_obj=func_obj if isinstance(func_obj, dict) else {},
                        dense_weight=args.dense_weight,
                    )
                else:
                    dense_combined = base_list  # Dense 불가 시 그대로

                # 3단계: 규칙 리랭크(프로젝트/확장/자카드)
                reranked = rule_rerank_candidates(
                    dense_combined, query_obj=func_obj if isinstance(func_obj, dict) else {},
                    bonus_same_project=args.bonus_same_project,
                    bonus_same_ext=args.bonus_same_ext,
                    bonus_token_jaccard=args.bonus_token_jaccard,
                )

                # 결과 구성(자기 자신/중복 제거)
                seen_ids = set()
                for bug_key, bug_info, score2 in reranked:
                    cand_id_val = candidate_id(bug_info, bug_key)
                    cand_public_id = public_candidate_id(bug_info, bug_key)
                    cand_dataset = candidate_dataset_label(bug_info)
                    if cand_id_val in current_ids:
                        continue
                    if cand_id_val in seen_ids:
                        continue
                    seen_ids.add(cand_id_val)

                    file_name = None
                    file_path = None
                    file_info = bug_info.get("file")
                    if isinstance(file_info, dict):
                        file_name = file_info.get("file_name")
                        file_path = file_info.get("file_path")

                    top_jsons.append({
                        "id": cand_id_val,
                        "public_id": cand_public_id,
                        "dataset": cand_dataset,
                        "corpus_key": str(bug_key),
                        "project_name": bug_info.get("project_name"),
                        "buggy_line_content": bug_info.get("buggy_line_content"),
                        "function": bug_info.get("function"),
                        "file_name": file_name,
                        "file_path": file_path,
                        "score_reranked": float(score2),  # 최종 점수
                    })
                    if len(top_jsons) >= args.topk:
                        break
            else:
                # 쿼리가 비었으면 빈 결과
                pass

            out_obj["bm25"] = {
                "code_line": query_text,
                "code_line_source": source,
                "expanded_tokens": tokenized_query,
                "topk": args.topk,
                "top": top_jsons,
                "params": {
                    "k1": args.bm25_k1, "b": args.bm25_b,
                    "weights": {
                        "w_buggy": args.w_buggy, "w_func": args.w_func,
                        "w_susp": args.w_susp, "w_reason": args.w_reason, "w_file": args.w_file
                    },
                    "stage_topm": args.stage_topm,
                    "no_expand": bool(args.no_expand),
                    "bonus_same_project": args.bonus_same_project,
                    "bonus_same_ext": args.bonus_same_ext,
                    "bonus_token_jaccard": args.bonus_token_jaccard,
                    "dense": {
                        "enabled": bool(args.dense_enable and dense_ready and dense_encoder is not None),
                        "model": args.dense_model,
                        "weight": args.dense_weight,
                        "cache": bundle.get("dense_cache_path", args.dense_cache),
                    },
                    "input_selection": {
                        "selected_input_file": str(real_path),
                        "selection_reason": str(real_input_reasons.get(str(real_path), "unknown")),
                        "all_bugs_files": [str(p) for p in selected_all_bug_inputs],
                        "all_bugs_selection_reasons": {
                            str(p): str(all_input_reasons.get(str(p), ""))
                            for p in selected_all_bug_inputs
                        },
                        "single_only": bool(args.single_only),
                        "same_dataset_only": bool(args.same_dataset_only),
                        "target_dataset": target_dataset,
                    },
                }
            }
            final_results[str(func_id)] = out_obj

        _save_json(out_path, final_results)
        print(f"✅ 완료: {real_path} -> {out_path} (총 {len(final_results)}개)")


if __name__ == "__main__":
    main()
