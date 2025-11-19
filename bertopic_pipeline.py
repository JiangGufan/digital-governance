# ---- temporary torch/transformers compatibility patch ----
import torch

if hasattr(torch.utils, "_pytree"):
    pt = torch.utils._pytree

    # 如果没有 register_pytree_node，但有 _register_pytree_node，就补一个 wrapper
    if not hasattr(pt, "register_pytree_node") and hasattr(pt, "_register_pytree_node"):
        _orig_register = pt._register_pytree_node

        def register_pytree_node(node_type, flatten_fn, unflatten_fn, *args, **kwargs):
            """
            兼容 transformers 新版的调用方式：
            - transformers 会传很多 keyword argument（serialized_type_name 等）
            - 老版本 torch 的 _register_pytree_node 只认前三个位置参数
            所以我们只把前三个位置参数传给原始函数，其他参数全部忽略。
            """
            return _orig_register(node_type, flatten_fn, unflatten_fn)

        pt.register_pytree_node = register_pytree_node
# ---- end of patch ----

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
# 其他 import ...


import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 配置区域：按需修改路径
# =========================
DATA_DIR = Path("/root//project/data/txt_doc_level")   # 143 个报告 txt 所在目录
REF_PATH = Path("/root//project/data/ref.txt")         # 可选的参考政策文本（如果没有就留空路径）
OUTPUT_DIR = Path("/root//project/data/output")
EMB_PATH = "/root/project/models/text2vec-base-chinese" # Sentence-BERT 模型路径（如果没有就用在线模型）

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "digital_attention_scores.csv"
MODEL_DIR = OUTPUT_DIR / "bertopic_model"

# 你提供的数字化治理相关关键词
DIGITAL_KEYWORDS = [
    "数据开放", "公共数据", "数据共享", "数据流通", "数据交易", "数据要素",
    "数据资源", "数据资产", "数据治理", "数据安全", "授权运营", "数据市场",
    "数据服务", "数据平台", "数据基础设施", "数据利用", "数据融合", "数据价值",
    "数据创新", "数字政府"
]


# =========================
# 1. 读入 143 份 txt 文档
# =========================
def load_documents(txt_dir: Path) -> Tuple[List[str], List[str]]:
    """
    从 txt_dir 读取所有 .txt 文件，返回：
    docs:   List[str]，每个元素是一份报告的全文
    names:  List[str]，与 docs 对应的文件名（不含路径）
    """
    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {txt_dir.resolve()}")

    docs, names = [], []
    for p in txt_files:
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            print(f"[WARN] empty txt file: {p.name}")
            continue
        docs.append(text)
        names.append(p.name)
    print(f"Loaded {len(docs)} documents from {txt_dir}")
    return docs, names


# =========================
# 2. 构建“数字化治理”参考向量
# =========================
def build_reference_text(keywords: List[str], ref_path: Path = None) -> str:
    """
    将关键词和可选的参考政策文本 ref.txt 拼在一起，
    形成一段较长的 reference text，用于编码成“数字化治理”参考向量。
    """
    ref_pieces = []

    # 把所有关键词拼成一段话
    ref_pieces.append(" ".join(keywords))

    # 如果提供了 ref.txt，就把内容也加入参考文本
    if ref_path is not None and ref_path.exists():
        ref_text = ref_path.read_text(encoding="utf-8", errors="ignore")
        ref_pieces.append(ref_text)
        print(f"Loaded reference policy text from {ref_path}")
    else:
        print("No ref.txt found or path not set. Using only keywords for reference embedding.")

    return "\n\n".join(ref_pieces)


def encode_reference(embedding_model, keywords: List[str], ref_path: Path) -> np.ndarray:
    """
    使用 Sentence-BERT 将 reference text 编码成一个向量。
    """
    ref_text = build_reference_text(keywords, ref_path)
    ref_emb = embedding_model.encode(ref_text, show_progress_bar=False)
    # 转成 shape=(1, dim) 便于后续 cosine_similarity
    return ref_emb.reshape(1, -1)


# =========================
# 3. 训练 BERTopic 模型
# =========================
def train_bertopic(docs: List[str]) -> Tuple[BERTopic, List[int], np.ndarray]:
    """
    用中文 Sentence-BERT + BERTopic 对 docs 做主题建模。
    返回：
      topic_model: 训练好的 BERTopic 模型
      topics:      每份文档的主主题 id 列表
      probs:       文档-主题 概率矩阵 (n_docs, n_topics)，可用于计算指数
    """
    print("Loading sentence-transformer embedding model...")
    embedding_model = SentenceTransformer(EMB_PATH) 

    print("Fitting BERTopic model...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="chinese",
        n_gram_range=(1, 3),
        min_topic_size=5,              # 视文档数量可调
        calculate_probabilities=True,  # 一定要 True 才有 probs
        verbose=True,
        low_memory=False,              # 文档不多，没必要开 low_memory
        nr_topics=None                 # 让模型自己决定主题数
    )

    topics, probs = topic_model.fit_transform(docs)
    print("Finished BERTopic training.")
    return topic_model, topics, probs


# =========================
# 4. 识别“数字化主题”
# =========================
def identify_digital_topics(
    topic_model: BERTopic,
    embedding_model,
    ref_emb: np.ndarray,
    keywords: List[str],
    sim_threshold: float = 0.35,
    top_n_words: int = 15
) -> Dict[int, Dict]:
    """
    用两种方式识别数字化主题：
      1) 语义相似度：topic 的代表词拼成字符串，和 ref_emb 余弦相似度 > sim_threshold
      2) 关键词命中：topic 的代表词中出现你的关键词

    返回：
      digital_topics: {topic_id: {"sim": float, "has_keyword": bool, "words": [str]}}
    """
    info = topic_model.get_topic_info()
    topic_ids = info["Topic"].tolist()

    # 将关键词做成集合，方便匹配
    kw_set = set(keywords)

    digital_topics = {}

    for tid in topic_ids:
        if tid == -1:
            # -1 通常是 outlier 主题，跳过
            continue

        words_scores = topic_model.get_topic(tid)  # List[(word, score)]
        if words_scores is None:
            continue

        # 提取前 top_n_words 个词
        words = [w for w, _ in words_scores[:top_n_words]]

        # 关键词命中：只要 topic 的词里有任意一个关键词即可
        has_keyword = any(w in kw_set for w in words)

        # 构造 topic_text，用于编码和 ref_emb 求余弦相似度
        topic_text = " ".join(words)
        topic_emb = embedding_model.encode(topic_text, show_progress_bar=False).reshape(1, -1)
        sim = float(cosine_similarity(ref_emb, topic_emb)[0, 0])

        # 判定是否为数字化主题
        if has_keyword or sim >= sim_threshold:
            digital_topics[tid] = {
                "sim": sim,
                "has_keyword": has_keyword,
                "words": words
            }

    # 按相似度打印一下识别结果，便于人工 sanity check
    print("\nIdentified digital-related topics:")
    for tid, info_t in sorted(digital_topics.items(), key=lambda x: -x[1]["sim"]):
        tag = []
        if info_t["has_keyword"]:
            tag.append("KW")
        if info_t["sim"] >= sim_threshold:
            tag.append("SIM")
        tag_str = "+".join(tag)
        preview = " ".join(info_t["words"][:8])
        print(f"  Topic {tid:>3}  [{tag_str}]  sim={info_t['sim']:.3f}  words: {preview}")

    return digital_topics


# =========================
# 5. 计算每个文档的数字化关注度指数
# =========================
def compute_digital_scores(
    probs: np.ndarray,
    doc_names: List[str],
    digital_topics: Dict[int, Dict],
) -> pd.DataFrame:
    """
    对每个文档，计算：
      - digital_index: 所有数字化主题概率之和（标量），反映整体关注度
      - digital_vector: 数字化主题概率组成的向量（可以做后续聚类/回归）

    返回一个 DataFrame，并在其中加入 digital_vector_* 的列。
    """
    n_docs, n_topics = probs.shape
    digital_topic_ids = sorted(digital_topics.keys())
    if not digital_topic_ids:
        raise ValueError("No digital topics identified; please adjust sim_threshold or keywords.")

    # 建立结果 DataFrame
    df = pd.DataFrame({
        "doc_name": doc_names,
    })

    # 取出所有数字化主题的概率子矩阵
    digital_probs = probs[:, digital_topic_ids]  # shape: (n_docs, n_digital_topics)

    # 标量指数：数字化主题概率之和（也可以换成加权和）
    digital_index = digital_probs.sum(axis=1)  # shape: (n_docs,)
    df["digital_index"] = digital_index

    # 也把每个数字化主题对应的概率单独展开成列
    for j, tid in enumerate(digital_topic_ids):
        col_name = f"topic_{tid}_prob"
        df[col_name] = digital_probs[:, j]

    print(f"\nComputed digital_index for {n_docs} documents.")
    print(f"Digital topics used: {digital_topic_ids}")
    print("digital_index range:", float(digital_index.min()), "to", float(digital_index.max()))
    return df


# =========================
# 6. 主流程
# =========================
def main():
    # 1) 读入 txt 文档
    docs, doc_names = load_documents(DATA_DIR)

    # 2) 加载 Sentence-BERT（和 BERTopic 的 embedding_model 保持一致）
    print("Loading sentence-transformer (for reference/topic encoding)...")
    embedding_model = SentenceTransformer(EMB_PATH)

    # 3) 构建“数字化治理”参考向量
    ref_emb = encode_reference(embedding_model, DIGITAL_KEYWORDS, REF_PATH)

    # 4) 训练 BERTopic 模型
    topic_model, topics, probs = train_bertopic(docs)

    # 5) 识别数字化主题
    digital_topics = identify_digital_topics(
        topic_model=topic_model,
        embedding_model=embedding_model,
        ref_emb=ref_emb,
        keywords=DIGITAL_KEYWORDS,
        sim_threshold=0.35,    # 如果识别太少/太多，可手动调高/调低
        top_n_words=15
    )

    # 6) 计算每份报告的数字化关注度指数 + 数字化主题向量
    df_scores = compute_digital_scores(
        probs=probs,
        doc_names=doc_names,
        digital_topics=digital_topics
    )

    # 7) 保存结果和模型
    df_scores.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved scores to: {OUTPUT_CSV}")

    topic_model.save(str(MODEL_DIR))
    print(f"Saved BERTopic model to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
