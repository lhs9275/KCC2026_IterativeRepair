import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


if os.environ.get("ESWA_DISABLE_TORCHVISION", "1") == "1":
    import importlib.util as _importlib_util

    _orig_find_spec = _importlib_util.find_spec

    def _find_spec(name, *args, **kwargs):
        if name == "torchvision" or name.startswith("torchvision."):
            return None
        return _orig_find_spec(name, *args, **kwargs)

    _importlib_util.find_spec = _find_spec

try:
    import torch
except Exception:
    torch = None  # type: ignore

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    _TRANSFORMERS_IMPORT_ERR: Optional[Exception] = exc
else:
    _TRANSFORMERS_IMPORT_ERR = None

try:
    from vllm import LLM, SamplingParams  # type: ignore
except Exception as exc:
    LLM = None  # type: ignore
    SamplingParams = Any  # type: ignore
    _VLLM_IMPORT_ERR: Optional[Exception] = exc
else:
    _VLLM_IMPORT_ERR = None


BACKEND_CHOICES = ("auto", "vllm", "hf")
HF_DEVICE_CHOICES = ("auto", "cpu", "cuda")
HF_DTYPE_CHOICES = ("auto", "float16", "bfloat16", "float32")


def torch_cuda_available() -> bool:
    try:
        return bool(torch is not None and torch.cuda.is_available())
    except Exception:
        return False


def _apply_stop_sequences(text: str, stop: Optional[List[str]]) -> str:
    out = str(text or "")
    if not stop:
        return out
    cut = len(out)
    for marker in stop:
        marker = str(marker or "")
        if not marker:
            continue
        idx = out.find(marker)
        if idx >= 0:
            cut = min(cut, idx)
    return out[:cut]


def _resolve_hf_device(hf_device: str) -> str:
    choice = str(hf_device or "auto").strip().lower()
    if choice not in HF_DEVICE_CHOICES:
        choice = "auto"
    if choice == "auto":
        return "cuda" if torch_cuda_available() else "cpu"
    if choice == "cuda" and not torch_cuda_available():
        raise RuntimeError("Requested --hf_device cuda but torch.cuda.is_available() is False.")
    return choice


def _resolve_hf_dtype(device: str, hf_dtype: str) -> Any:
    if torch is None:
        return None
    choice = str(hf_dtype or "auto").strip().lower()
    if choice not in HF_DTYPE_CHOICES:
        choice = "auto"
    if choice == "float16":
        return torch.float16
    if choice == "bfloat16":
        return torch.bfloat16
    if choice == "float32":
        return torch.float32
    if device == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def _format_prompt(tokenizer: Any, prompt: str, system_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return str(prompt or "")


def _truncate_max_length(tokenizer: Any, max_input_tokens: Optional[int], fallback: int) -> Optional[int]:
    requested = int(max_input_tokens or fallback or 0)
    if requested <= 0:
        return None
    model_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_limit, int) and 0 < model_limit < 1_000_000:
        return min(requested, int(model_limit))
    return requested


def build_sampling_params(kwargs: Dict[str, Any]) -> Any:
    if SamplingParams is None:
        raise RuntimeError(f"SamplingParams is unavailable (vLLM import failed: {_VLLM_IMPORT_ERR})")
    params_kwargs = dict(kwargs or {})
    for _ in range(16):
        try:
            return SamplingParams(**params_kwargs)
        except TypeError as exc:
            msg = str(exc)
            marker = "unexpected keyword argument '"
            if marker not in msg:
                if "seed" in params_kwargs:
                    params_kwargs.pop("seed", None)
                    continue
                raise
            bad = msg.split(marker, 1)[1].split("'", 1)[0]
            if bad in params_kwargs:
                params_kwargs.pop(bad, None)
                continue
            raise
    return SamplingParams(**params_kwargs)


@dataclass
class GenerationRecord:
    text: str
    tokens_in: int
    tokens_out: int


class LLMBackend:
    backend_name = "base"

    def __init__(self, model_name: str, system_prompt: str = ""):
        self.model_name = str(model_name)
        self.system_prompt = str(system_prompt or "").strip()
        self.metrics: Dict[str, float] = {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "batches": 0.0,
            "max_gpu_mb": 0.0,
        }

    def generate(
        self,
        prompts: List[str],
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        records = self.generate_records(
            prompts,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
            stop=stop,
            **kwargs,
        )
        return [record.text for record in records]

    def generate_records(
        self,
        prompts: List[str],
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[GenerationRecord]:
        raise NotImplementedError


class VLLMBackend(LLMBackend):
    backend_name = "vllm"

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
    ):
        super().__init__(model_name=model_name, system_prompt=system_prompt)
        if LLM is None:
            raise RuntimeError(
                "vLLM import failed. Install compatible vllm/torch packages first. "
                f"Original error: {_VLLM_IMPORT_ERR}"
            )
        self._ensure_vllm_device()
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=int(max_model_len),
        )
        self.tokenizer = self.llm.get_tokenizer()

    @staticmethod
    def _ensure_vllm_device() -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not available in this environment.")
        try:
            from vllm.platforms import current_platform
        except Exception as exc:
            raise RuntimeError(f"Failed to import vLLM platforms for device detection: {exc}") from exc
        if getattr(current_platform, "device_type", ""):
            return
        if torch_cuda_available():
            try:
                from vllm.platforms.cuda import CudaPlatform
                import vllm.platforms as platforms

                platforms.current_platform = CudaPlatform()
                if getattr(platforms.current_platform, "device_type", ""):
                    logging.warning(
                        "vLLM platform detection failed; forcing CUDA because torch.cuda.is_available() is True."
                    )
                    return
            except Exception as exc:
                logging.warning("Failed to force CUDA platform: %s", exc)
        raise RuntimeError(
            "vLLM could not infer a device. "
            f"torch.cuda.is_available()={torch_cuda_available()}, "
            f"torch.version.cuda={getattr(torch, 'version', None).cuda if getattr(torch, 'version', None) else None}. "
            "Install CUDA-enabled torch/vLLM or use HF fallback."
        )

    def _gpu_mem_mb(self) -> float:
        try:
            if torch_cuda_available():
                return float(torch.cuda.memory_allocated() / (1024 * 1024))
        except Exception:
            pass
        return 0.0

    def generate_records(
        self,
        prompts: List[str],
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[GenerationRecord]:
        formatted_prompts = [
            _format_prompt(self.tokenizer, prompt, self.system_prompt) for prompt in list(prompts or [])
        ]
        # Truncate prompts that exceed max_model_len - max_new_tokens
        _max_prompt_tokens = int(self.llm.llm_engine.model_config.max_model_len) - int(max_new_tokens)
        truncated = []
        for fp in formatted_prompts:
            ids = self.tokenizer.encode(fp)
            if len(ids) > _max_prompt_tokens:
                ids = ids[-_max_prompt_tokens:]
                fp = self.tokenizer.decode(ids, skip_special_tokens=False)
            truncated.append(fp)
        formatted_prompts = truncated
        do_sample = bool(kwargs.get("do_sample", True))
        n = max(1, int(kwargs.get("n", 1) or 1))
        if not do_sample:
            temperature = 0.0
            top_p = 1.0
            n = 1
        # vLLM rejects n>1 with greedy sampling (temperature=0).
        # When caller wants multiple candidates, switch to near-greedy sampling
        # (tiny temperature) instead of forcing n=1, which would cripple iter 1.
        if float(temperature) <= 0.0 and n > 1:
            temperature = 0.01
        sampling_kwargs: Dict[str, Any] = {
            "n": n,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "repetition_penalty": float(kwargs.get("repetition_penalty", 1.05)),
            "max_tokens": int(max_new_tokens),
            "seed": int(seed if seed is not None else random.randint(0, 10_000_000)),
        }
        top_k = int(kwargs.get("top_k", 0) or 0)
        if top_k > 0:
            sampling_kwargs["top_k"] = top_k
        presence_penalty = float(kwargs.get("presence_penalty", 0.0) or 0.0)
        if presence_penalty != 0.0:
            sampling_kwargs["presence_penalty"] = presence_penalty
        frequency_penalty = float(kwargs.get("frequency_penalty", 0.0) or 0.0)
        if frequency_penalty != 0.0:
            sampling_kwargs["frequency_penalty"] = frequency_penalty
        if stop:
            sampling_kwargs["stop"] = list(stop)

        sampling_params = build_sampling_params(sampling_kwargs)
        raw_outputs = self.llm.generate(formatted_prompts, sampling_params)

        records: List[GenerationRecord] = []
        input_tokens_sum = 0
        output_tokens_sum = 0
        for req in raw_outputs:
            candidates = (req.outputs or [])[:n]
            in_tok = len(getattr(req, "prompt_token_ids", []) or [])
            input_tokens_sum += int(in_tok)
            if not candidates:
                records.append(GenerationRecord(text="", tokens_in=int(in_tok), tokens_out=0))
                continue
            for idx, cand in enumerate(candidates):
                out_tok = len(getattr(cand, "token_ids", []) or [])
                output_tokens_sum += int(out_tok)
                records.append(
                    GenerationRecord(
                        text=_apply_stop_sequences(cand.text if cand is not None else "", stop),
                        tokens_in=int(in_tok if idx == 0 else 0),
                        tokens_out=int(out_tok),
                    )
                )

        self.metrics["max_gpu_mb"] = max(self.metrics.get("max_gpu_mb", 0.0), self._gpu_mem_mb())
        self.metrics["batches"] = self.metrics.get("batches", 0.0) + 1
        self.metrics["input_tokens"] = self.metrics.get("input_tokens", 0.0) + input_tokens_sum
        self.metrics["output_tokens"] = self.metrics.get("output_tokens", 0.0) + output_tokens_sum
        return records


class HFBackend(LLMBackend):
    backend_name = "hf"

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        hf_device: str = "auto",
        hf_dtype: str = "auto",
        max_model_len: int = 8192,
    ):
        super().__init__(model_name=model_name, system_prompt=system_prompt)
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError(
                "transformers import failed. Install compatible transformers/torch packages first. "
                f"Original error: {_TRANSFORMERS_IMPORT_ERR}"
            )
        self.device = _resolve_hf_device(hf_device)
        self.dtype = _resolve_hf_dtype(self.device, hf_dtype)
        self.max_model_len = int(max_model_len)

        tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.dtype is not None:
            model_kwargs["torch_dtype"] = self.dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        if self.device == "cuda":
            self.model.to("cuda")

    def _gpu_mem_mb(self) -> float:
        try:
            if self.device == "cuda" and torch_cuda_available():
                return float(torch.cuda.memory_allocated() / (1024 * 1024))
        except Exception:
            pass
        return 0.0

    def generate_records(
        self,
        prompts: List[str],
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[GenerationRecord]:
        if torch is None:
            raise RuntimeError("PyTorch is required for HF backend.")

        do_sample = bool(kwargs.get("do_sample", True))
        n = max(1, int(kwargs.get("n", 1) or 1))
        if not do_sample:
            temperature = 0.0
            top_p = 1.0
            n = 1

        records: List[GenerationRecord] = []
        input_tokens_sum = 0
        output_tokens_sum = 0
        max_input_tokens = _truncate_max_length(self.tokenizer, kwargs.get("max_input_tokens"), self.max_model_len)
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "num_return_sequences": int(n),
            "repetition_penalty": float(kwargs.get("repetition_penalty", 1.05)),
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        if do_sample:
            generation_kwargs["temperature"] = max(0.01, float(temperature))
            generation_kwargs["top_p"] = float(top_p)
            top_k = int(kwargs.get("top_k", 0) or 0)
            if top_k > 0:
                generation_kwargs["top_k"] = top_k

        for prompt in list(prompts or []):
            if seed is not None:
                torch.manual_seed(int(seed))
                if self.device == "cuda" and torch_cuda_available():
                    torch.cuda.manual_seed_all(int(seed))

            prompt_text = _format_prompt(self.tokenizer, prompt, self.system_prompt)
            tokenized = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=bool(max_input_tokens),
                max_length=max_input_tokens,
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            in_tok = int(tokenized["input_ids"].shape[-1])

            with torch.inference_mode():
                output_ids = self.model.generate(**tokenized, **generation_kwargs)

            input_tokens_sum += in_tok
            for idx in range(output_ids.shape[0]):
                seq = output_ids[idx]
                gen_ids = seq[in_tok:]
                out_tok = int(gen_ids.shape[-1])
                output_tokens_sum += out_tok
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                records.append(
                    GenerationRecord(
                        text=_apply_stop_sequences(text, stop),
                        tokens_in=int(in_tok if idx == 0 else 0),
                        tokens_out=out_tok,
                    )
                )

        self.metrics["max_gpu_mb"] = max(self.metrics.get("max_gpu_mb", 0.0), self._gpu_mem_mb())
        self.metrics["batches"] = self.metrics.get("batches", 0.0) + 1
        self.metrics["input_tokens"] = self.metrics.get("input_tokens", 0.0) + input_tokens_sum
        self.metrics["output_tokens"] = self.metrics.get("output_tokens", 0.0) + output_tokens_sum
        return records


def create_backend(
    *,
    backend: str,
    model_name: str,
    system_prompt: str = "",
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.85,
    hf_model_name: Optional[str] = None,
    hf_device: str = "auto",
    hf_dtype: str = "auto",
    logger: Optional[Any] = None,
) -> Tuple[LLMBackend, str]:
    backend_req = str(backend or "auto").strip().lower()
    if backend_req not in BACKEND_CHOICES:
        backend_req = "auto"
    log = logger or logging
    hf_model = str(hf_model_name or model_name)

    if backend_req == "vllm":
        instance = VLLMBackend(
            model_name=model_name,
            system_prompt=system_prompt,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        return instance, instance.backend_name

    if backend_req == "hf":
        instance = HFBackend(
            model_name=hf_model,
            system_prompt=system_prompt,
            hf_device=hf_device,
            hf_dtype=hf_dtype,
            max_model_len=max_model_len,
        )
        return instance, instance.backend_name

    if not torch_cuda_available():
        log.warning("Falling back to HF backend because CUDA unavailable.")
        instance = HFBackend(
            model_name=hf_model,
            system_prompt=system_prompt,
            hf_device=hf_device,
            hf_dtype=hf_dtype,
            max_model_len=max_model_len,
        )
        return instance, instance.backend_name

    try:
        instance = VLLMBackend(
            model_name=model_name,
            system_prompt=system_prompt,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        return instance, instance.backend_name
    except Exception as exc:
        log.warning("Falling back to HF backend because vLLM init failed: %s", exc)
        instance = HFBackend(
            model_name=hf_model,
            system_prompt=system_prompt,
            hf_device=hf_device,
            hf_dtype=hf_dtype,
            max_model_len=max_model_len,
        )
        return instance, instance.backend_name


def describe_backend_state() -> Dict[str, Any]:
    return {
        "torch_available": bool(torch is not None),
        "torch_cuda_available": bool(torch_cuda_available()),
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None) if torch is not None else None,
        "vllm_import_error": str(_VLLM_IMPORT_ERR) if _VLLM_IMPORT_ERR else "",
        "transformers_import_error": str(_TRANSFORMERS_IMPORT_ERR) if _TRANSFORMERS_IMPORT_ERR else "",
        "backend_choices": list(BACKEND_CHOICES),
        "hf_device_choices": list(HF_DEVICE_CHOICES),
        "hf_dtype_choices": list(HF_DTYPE_CHOICES),
    }
