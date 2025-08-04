# -*- coding: utf-8 -*-
import os
import streamlit as st
import pandas as pd
import joblib

# 작은 머신에서 과도한 스레드로 느려지는 것 방지
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

def _file_sig(path: str):
    """파일 변경 시 캐시가 자연히 갱신되도록 (경로+크기+mtime) 키 생성"""
    try:
        stt = os.stat(path)
        return (path, stt.st_size, int(stt.st_mtime))
    except FileNotFoundError:
        return (path, None, None)

@st.cache_data(ttl=0, max_entries=16, show_spinner=False)
def load_csv(
    path: str,
    usecols=None,
    dtype=None,
    nrows=None,
    encoding=None,
    memory_map: bool = True,
    arrow: bool = True,
    low_memory: bool = False,
    **kwargs,
):
    """
    빠른 CSV 로더 (캐시). 주요 가속 포인트:
    - PyArrow 엔진 사용(있으면) → 파싱 속도/메모리 효율 향상
    - usecols/dtype/nrows로 불필요한 파싱 최소화
    - memory_map으로 I/O 오버헤드 완화
    - low_memory=False로 타입 추론 일괄화(속도 ↑, 메모리 ↑)
    """
    # 파일 변경 감지용 키를 인자로 섞어 캐시 무효화
    _ = _file_sig(path)

    engine = None
    if arrow:
        try:
            import pyarrow  # noqa: F401
            engine = "pyarrow"
        except Exception:
            engine = None

    # dtype_backend='pyarrow'는 pandas 2.x + pyarrow에서 유효
    kw = dict(
        usecols=usecols,
        dtype=dtype,
        nrows=nrows,
        encoding=encoding,
        memory_map=memory_map,
        low_memory=low_memory,
        **kwargs,
    )
    if engine:
        kw["engine"] = engine
        # pandas>=2.0에서만 지원. 가능하면 더 빠르고 메모리 효율적
        if hasattr(pd, "options"):
            kw["dtype_backend"] = "pyarrow"

    return pd.read_csv(path, **{k: v for k, v in kw.items() if v is not None})

@st.cache_resource(show_spinner=False)
def load_pickle(path: str, mmap: bool = True):
    """
    빠른 pickle/joblib 로더 (캐시). 큰 배열 포함 모델은 mmap_mode='r'로 즉시 사용.
    파일 변경 시 캐시 무효화를 위해 시그니처를 키에 포함.
    """
    _ = _file_sig(path)
    mmap_mode = "r" if mmap else None
    return joblib.load(path, mmap_mode=mmap_mode)
