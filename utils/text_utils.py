import csv
import os
import numpy as np
import torch

# ===============================
# Sentence-transformers
# ===============================
from sentence_transformers import SentenceTransformer

# ===============================
# Optional external models
# ===============================
try:
    from transformers import AutoTokenizer, AutoModel
    _HAS_QWEN = True
except Exception:
    _HAS_QWEN = False

try:
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


# ============================================================
# GLOBAL MODEL CACHES
# ============================================================
_MINILM = None
_E5 = None
_SGPT = None
_QWEN = None


# ============================================================
# 1) MiniLM baseline
# ============================================================
def _get_minilm():
    global _MINILM
    if _MINILM is None:
        _MINILM = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MINILM


# ============================================================
# 2) E5-large
# ============================================================
def _get_e5():
    global _E5
    if _E5 is None:
        _E5 = SentenceTransformer("intfloat/e5-large")
    return _E5


# ============================================================
# 3) SGPT-text
# ============================================================
def _get_sgpt():
    global _SGPT
    if _SGPT is None:
        _SGPT = SentenceTransformer("Muennighoff/SGPT-1.3B")
    return _SGPT


# ============================================================
# 4) Qwen-Audio text encoder
# ============================================================
def _get_qwen():
    global _QWEN
    if _QWEN is None:
        if not _HAS_QWEN:
            raise RuntimeError("Qwen-Audio를 사용하려면 transformers 설치 필요!")
        _QWEN = AutoModel.from_pretrained(
            "Qwen/Qwen-Audio",
            trust_remote_code=True
        ).eval()
    return _QWEN


def _encode_qwen(texts):
    model = _get_qwen()
    with torch.no_grad():
        outputs = model(None, texts=texts, return_dict=True)
        return outputs["text_embeds"].cpu().numpy()


# ============================================================
# 5) OpenAI embedding model
# ============================================================
def _encode_openai(texts):
    if not _HAS_OPENAI:
        raise RuntimeError("OpenAI API 사용하려면 openai 패키지 필요")
    out = []
    for t in texts:
        emb = openai.embeddings.create(
            model="text-embedding-3-large",
            input=t
        )["data"][0]["embedding"]
        out.append(np.array(emb))
    return np.stack(out, axis=0)


# ============================================================
# Demucs strict keyword routing  (substring only)
# ============================================================
_DEMUCS_KEYWORDS = {
    "vocals": "vocals",
    "vocal": "vocals",
    "voice": "vocals",
    "singing": "vocals",
    "singer": "vocals",

    "drum": "drums",
    "drums": "drums",
    "snare": "drums",
    "kick": "drums",
    "percussion": "drums",

    "bass": "bass",
    "bass guitar": "bass",
}


# ============================================================
# AudioSet labels
# ============================================================
def load_audioset_labels():
    meta_path = "./audioset_tagging_cnn/metadata/class_labels_indices.csv"
    if not os.path.exists(meta_path):
        return None
    labels = []
    with open(meta_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["display_name"])
    return labels


# ============================================================
# Encode labels/prompt with selected model
# ============================================================
def _encode_texts(texts, model_type, clap_encoder=None):
    """
    texts: list[str]
    model_type: "minilm" / "clap" / "e5" / "sgpt" / "qwen" / "openai"
    """

    if model_type == "minilm":
        model = _get_minilm()
        return model.encode(texts, normalize_embeddings=True)

    elif model_type == "e5":
        model = _get_e5()
        return model.encode(texts, normalize_embeddings=True)

    elif model_type == "sgpt":
        model = _get_sgpt()
        return model.encode(texts, normalize_embeddings=True)

    elif model_type == "clap":
        if clap_encoder is None:
            raise ValueError("CLAP text embedding을 사용하려면 clap_encoder 전달 필요!")
        with torch.no_grad():
            return clap_encoder._get_text_embed(texts).cpu().numpy()

    elif model_type == "qwen":
        return _encode_qwen(texts)

    elif model_type == "openai":
        return _encode_openai(texts)

    else:
        raise ValueError(f"Unknown embedding model: {model_type}")


# ============================================================
# SED label selection
# ============================================================
def select_target_class(
    labels,
    text_prompt,
    embedding_model_type="minilm",
    clap_encoder=None,
):
    """
    여러 프롬프트 처리:
    "dog barking, engine idling" → ["dog barking", "engine idling"]
    """

    if labels is None:
        return None

    sub_prompts = [p.strip() for p in text_prompt.split(",") if p.strip()]

    # 단일 프롬프트 처리
    if len(sub_prompts) == 1:
        return _select_single(
            labels,
            sub_prompts[0],
            embedding_model_type,
            clap_encoder,
        )

    # 복수 프롬프트 처리
    out = []
    for sp in sub_prompts:
        t = _select_single(labels, sp, embedding_model_type, clap_encoder)
        if t is not None:
            out.append(t)

    if len(out) == 0:
        return None

    return out


# ============================================================
# 내부 단일 프롬프트 매핑 함수
# ============================================================
def _select_single(labels, text_prompt, embedding_model_type, clap_encoder):
    t = text_prompt.lower().strip()

    # ----------------------------------------------------
    # 1) Strict substring Demucs routing ONLY
    # ----------------------------------------------------
    for key, value in _DEMUCS_KEYWORDS.items():
        if key in t:
            return {"demucs_target": value}

    # ----------------------------------------------------
    # 2) AudioSet exact match
    # ----------------------------------------------------
    for i, name in enumerate(labels):
        if name.lower() == t:
            return i

    # ----------------------------------------------------
    # 3) substring match (label includes prompt or vice versa)
    # ----------------------------------------------------
    for i, name in enumerate(labels):
        n = name.lower()
        if t in n or n in t:
            return i

    # ----------------------------------------------------
    # 4) embedding-based fallback (SED only)
    # ----------------------------------------------------
    label_emb = _encode_texts(labels, embedding_model_type, clap_encoder)
    q_emb = _encode_texts([text_prompt], embedding_model_type, clap_encoder)[0]

    sim = label_emb @ q_emb
    return int(np.argmax(sim))
