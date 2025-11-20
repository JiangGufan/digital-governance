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
REF_PATH = Path("/root//project/data/ref2.txt")         # 可选的参考政策文本（如果没有就留空路径）
OUTPUT_DIR = Path("/root//project/output")
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

import re


def split_sentences_cn(text: str) -> List[str]:
    """
    非严格的中文句子切分：按。！？；以及换行拆分。
    """
    # 先统一换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 用标点拆分
    parts = re.split(r"[。！？；\n]+", text)
    # 去掉空白
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences


def build_focus_docs_and_kw_density(
    docs: List[str],
    keywords: List[str],
    extra_char_triggers: str = "数"
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    对每篇原始报告 text_i：
      1) 切成句子；
      2) 挑出“疑似数字化相关”的句子：
           - 包含 DIGITAL_KEYWORDS 中任一，
           - 或者包含 extra_char_triggers（默认 '数'，捕捉“数字化、数据、数智”等）；
      3) 用这些句子拼成 focus_text_i（如果一篇报告完全没有，就回退为全文）；
      4) 同时统计关键词总出现次数 kw_count_i，并计算 kw_density_i = kw_count_i / len(text_i)。

    返回：
      focus_docs: List[str]，用于后续做 doc-ref 语义相似度；
      kw_count:   np.ndarray 形状 [n_docs]；
      kw_density: np.ndarray 形状 [n_docs]，已经按原文长度归一。
    """
    focus_docs = []
    kw_count_list = []
    kw_density_list = []

    for text in docs:
        sentences = split_sentences_cn(text)

        digital_sentences = []
        kw_count = 0

        for sent in sentences:
            hit_kw = False
            # 精确匹配关键词
            for kw in keywords:
                c = sent.count(kw)
                if c > 0:
                    kw_count += c
                    hit_kw = True

            # 粗触发：包含“数”这个字（数字化、数据、数智、数字经济…）
            if (not hit_kw) and any(ch in sent for ch in extra_char_triggers):
                digital_sentences.append(sent)
            elif hit_kw:
                digital_sentences.append(sent)

        # 构造 focus_text：如果有数字化相关句子，就只用这些；否则退回全文
        if digital_sentences:
            focus_text = "。".join(digital_sentences)
        else:
            focus_text = text

        focus_docs.append(focus_text)

        # 关键词密度：按原文长度归一
        text_len = max(len(text), 1)
        kw_density = kw_count / text_len

        kw_count_list.append(kw_count)
        kw_density_list.append(kw_density)

    kw_count_arr = np.array(kw_count_list, dtype=float)
    kw_density_arr = np.array(kw_density_list, dtype=float)

    print("\nKeyword statistics:")
    print("  total kw_count range:", float(kw_count_arr.min()), "to", float(kw_count_arr.max()))
    print("  kw_density range:", float(kw_density_arr.min()), "to", float(kw_density_arr.max()))

    return focus_docs, kw_count_arr, kw_density_arr


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
# 4+. 识别“数字化主题”，更严格版
# =========================

def identify_digital_topics_strict(
    topic_model: BERTopic,
    embedding_model,
    ref_emb: np.ndarray,
    keywords: List[str],
    top_n_words: int = 50
) -> Dict[int, Dict]:
    """
    严格版：只把“词表里真的出现数据/数字相关词”的主题当成数字化主题。
    逻辑：
      1) 从 BERTopic 取每个主题的前 top_n_words 个代表词；
      2) 如果这些词里包含 DIGITAL_KEYWORDS，或者至少包含“数据”/“数字”任一，则认为是候选数字化主题；
      3) 再算一下和 ref_emb 的语义相似度（用于排序和人工检查，不作为硬门槛）。
    返回:
      {topic_id: {"sim": float, "kw_hits": [kw...], "words": [w1,w2,...]}}
    """
    info = topic_model.get_topic_info()
    topic_ids = info["Topic"].tolist()

    digital_topics = {}

    for tid in topic_ids:
        if tid == -1:
            # -1 是 outlier 主题，忽略
            continue

        words_scores = topic_model.get_topic(tid)
        if not words_scores:
            continue

        # 取前 top_n_words 个代表词
        words = [w for w, _ in words_scores[:top_n_words]]
        joined = "".join(words)

        # 精确匹配你给的 DIGITAL_KEYWORDS
        kw_hits = [kw for kw in keywords if kw in joined]

        # 粗匹配：至少出现“数据”或“数字”其中之一
        coarse_hit = ("数据" in joined) or ("数字" in joined)

        if not kw_hits and not coarse_hit:
            # 完全看不到数字化痕迹，就不是数字化主题
            continue

        # 计算一下该主题与 ref 的语义相似度（用于排序和诊断）
        topic_text = " ".join(words)
        topic_emb = embedding_model.encode(topic_text, show_progress_bar=False).reshape(1, -1)
        sim = float(cosine_similarity(ref_emb, topic_emb)[0, 0])

        digital_topics[tid] = {
            "sim": sim,
            "kw_hits": kw_hits,
            "words": words
        }

    # 打印一下，方便你人工看主题是否真和“数据/数字”有关
    print("\n[Strict] Identified digital-related topics:")
    for tid, info_t in sorted(digital_topics.items(), key=lambda x: -x[1]["sim"]):
        kw_str = ",".join(info_t["kw_hits"]) if info_t["kw_hits"] else "粗匹配(含'数据'或'数字')"
        preview = " ".join(info_t["words"][:10])
        print(f"  Topic {tid:>3}  sim={info_t['sim']:.3f}  kw=[{kw_str}]  words: {preview}")

    return digital_topics


# =========================
# 5. 计算每个文档的数字化关注度指数
# =========================
# def compute_digital_scores(
#     probs: np.ndarray,
#     doc_names: List[str],
#     digital_topics: Dict[int, Dict],
# ) -> pd.DataFrame:
#     """
#     对每个文档，计算：
#       - digital_index: 所有数字化主题概率之和（标量），反映整体关注度
#       - digital_vector: 数字化主题概率组成的向量（可以做后续聚类/回归）

#     返回一个 DataFrame，并在其中加入 digital_vector_* 的列。
#     """
#     n_docs, n_topics = probs.shape
#     digital_topic_ids = sorted(digital_topics.keys())
#     if not digital_topic_ids:
#         raise ValueError("No digital topics identified; please adjust sim_threshold or keywords.")

#     # 建立结果 DataFrame
#     df = pd.DataFrame({
#         "doc_name": doc_names,
#     })

#     # 取出所有数字化主题的概率子矩阵
#     digital_probs = probs[:, digital_topic_ids]  # shape: (n_docs, n_digital_topics)

#     # 标量指数：数字化主题概率之和（也可以换成加权和）
#     digital_index = digital_probs.sum(axis=1)  # shape: (n_docs,)
#     df["digital_index"] = digital_index

#     # 也把每个数字化主题对应的概率单独展开成列
#     for j, tid in enumerate(digital_topic_ids):
#         col_name = f"topic_{tid}_prob"
#         df[col_name] = digital_probs[:, j]

#     print(f"\nComputed digital_index for {n_docs} documents.")
#     print(f"Digital topics used: {digital_topic_ids}")
#     print("digital_index range:", float(digital_index.min()), "to", float(digital_index.max()))
#     return df

def compute_digital_scores(
    probs: np.ndarray,
    doc_names: List[str],
    digital_topics: Dict[int, Dict],
    doc_sem_sim: np.ndarray,
    doc_sem_sim_norm: np.ndarray,
    kw_density_norm: np.ndarray,
) -> pd.DataFrame:
    """
    对每个文档，计算：
      - topic_index: 纯基于“数字化主题概率”的指数（如果有数字化主题的话）
      - doc_sem_sim: 文档与 ref 的余弦相似度（原始值）
      - doc_sem_sim_norm: 上面那个归一化到 [0,1]
      - digital_index_combined:
            若有数字化主题: 0.5 * topic_index_norm + 0.5 * doc_sem_sim_norm
            若没有数字化主题: == doc_sem_sim_norm

    其中，topic_index_norm 是对 topic_index 做 min-max 归一后的结果。
    """
    n_docs, n_topics = probs.shape
    digital_topic_ids = sorted(digital_topics.keys())

    df = pd.DataFrame({
        "doc_name": doc_names,
    })

    # 文档层语义相似度（这部分无论如何都存在）
    df["doc_sem_sim"] = doc_sem_sim
    df["doc_sem_sim_norm"] = doc_sem_sim_norm

    # ========== 情况 A：有识别出的数字化主题 ==========
    if digital_topic_ids:
        print(f"\n[compute_digital_scores] Using {len(digital_topic_ids)} digital topics:", digital_topic_ids)
        # 取出数字化主题的概率子矩阵
        digital_probs = probs[:, digital_topic_ids]  # shape: (n_docs, n_digital_topics)

        # 主题层指数：数字化主题概率之和
        topic_index = digital_probs.sum(axis=1)  # shape: (n_docs,)
        df["topic_index"] = topic_index

        # 每个数字化主题的概率单独展开出来
        for j, tid in enumerate(digital_topic_ids):
            col_name = f"topic_{tid}_prob"
            df[col_name] = digital_probs[:, j]

        # 对 topic_index 做 min-max 归一
        min_t, max_t = topic_index.min(), topic_index.max()
        if max_t > min_t:
            topic_index_norm = (topic_index - min_t) / (max_t - min_t)
        else:
            topic_index_norm = np.zeros_like(topic_index)

        df["topic_index_norm"] = topic_index_norm

        # 综合指数 = 主题层 + 文档层 的简单平均
        df["digital_index_combined"] = 0.5 * topic_index_norm + 0.5 * doc_sem_sim_norm

        print(f"\nComputed scores for {n_docs} documents.")
        print("topic_index range:", float(topic_index.min()), "to", float(topic_index.max()))
        print("digital_index_combined range:",
              float(df['digital_index_combined'].min()),
              "to", float(df['digital_index_combined'].max()))
        return df

    # ========== 情况 B：没有任何数字化主题 ==========
    print("\n[compute_digital_scores] WARNING: No digital topics identified.")
    print("=> 当前使用 doc_sem_sim_norm 和 kw_density_norm 共同作为数字化治理关联度指数。")

    df["topic_index"] = np.zeros(n_docs, dtype=float)
    df["topic_index_norm"] = np.zeros(n_docs, dtype=float)

    # 把关键词密度也存进去方便你后续分析
    df["kw_density_norm"] = kw_density_norm

    # 综合指数：语义相似度 + 关键词密度 各占一半
    df["digital_index_combined"] = 0.5 * doc_sem_sim_norm + 0.5 * kw_density_norm

    print(f"\nComputed scores for {n_docs} documents (doc_sem_sim + kw_density).")
    print("digital_index_combined range:",
          float(df['digital_index_combined'].min()),
          "to", float(df['digital_index_combined'].max()))
    return df


# =========================
# 6. 计算每份报告与 ref 的语义相似度
# =========================
def compute_doc_semantic_scores(
    docs: List[str],
    embedding_model,
    ref_emb: np.ndarray,
    batch_size: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算每篇文档与 ref 的语义相似度：
      - raw_sims: 原始余弦相似度 (形状: [n_docs])
      - sims_norm: 经 min-max 归一后的相似度 (0~1)
    """
    print("\nComputing document-ref semantic similarity...")
    doc_embs = embedding_model.encode(docs, batch_size=batch_size, show_progress_bar=True)
    raw_sims = cosine_similarity(doc_embs, ref_emb).reshape(-1)

    min_s, max_s = raw_sims.min(), raw_sims.max()
    if max_s > min_s:
        sims_norm = (raw_sims - min_s) / (max_s - min_s)
    else:
        sims_norm = np.zeros_like(raw_sims)

    print("Doc-ref similarity range:", float(raw_sims.min()), "to", float(raw_sims.max()))
    return raw_sims, sims_norm


# =========================
# 7. 主流程
# =========================
def main():
    # 1) 读入 txt 文档
    docs, doc_names = load_documents(DATA_DIR)

    # 1.5) 基于 DIGITAL_KEYWORDS 构造“数字化焦点文本” + 关键词密度
    focus_docs, kw_count, kw_density = build_focus_docs_and_kw_density(
        docs,
        DIGITAL_KEYWORDS,
        extra_char_triggers="数"
    )

    # 对 kw_density 做 min-max 归一
    min_k, max_k = kw_density.min(), kw_density.max()
    if max_k > min_k:
        kw_density_norm = (kw_density - min_k) / (max_k - min_k)
    else:
        kw_density_norm = np.zeros_like(kw_density)

    # 2) 加载 Sentence-BERT（和 BERTopic 的 embedding_model 保持一致）
    print("Loading sentence-transformer (for reference/topic encoding)...")
    embedding_model = SentenceTransformer(EMB_PATH)

    # 3) 构建“数字化治理”参考向量
    ref_emb = encode_reference(embedding_model, DIGITAL_KEYWORDS, REF_PATH)
    # 3.5) 每篇报告与 ref 的语义相似度
    doc_sem_sim, doc_sem_sim_norm = compute_doc_semantic_scores(docs, embedding_model, ref_emb)


    # 4) 训练 BERTopic 模型
    topic_model, topics, probs = train_bertopic(docs)

    # # 5) 识别数字化主题
    # digital_topics = identify_digital_topics(
    #     topic_model=topic_model,
    #     embedding_model=embedding_model,
    #     ref_emb=ref_emb,
    #     keywords=DIGITAL_KEYWORDS,
    #     sim_threshold=0.35,    # 如果识别太少/太多，可手动调高/调低
    #     top_n_words=15
    # )

    # 5) 严格识别数字化主题（必须包含“数据/数字”等）
    digital_topics = identify_digital_topics_strict(
        topic_model=topic_model,
        embedding_model=embedding_model,
        ref_emb=ref_emb,
        keywords=DIGITAL_KEYWORDS,
        top_n_words=50
    )

    # # 6) 计算每份报告的数字化关注度指数 + 数字化主题向量
    # df_scores = compute_digital_scores(
    #     probs=probs,
    #     doc_names=doc_names,
    #     digital_topics=digital_topics
    # )
    # 6) 计算每份报告的数字化关注度指数 (新版)
    df_scores = compute_digital_scores(
        probs=probs,
        doc_names=doc_names,
        digital_topics=digital_topics,
        doc_sem_sim=doc_sem_sim,
        doc_sem_sim_norm=doc_sem_sim_norm,
        kw_density_norm=kw_density_norm
    )


    # 7) 保存结果和模型
    df_scores.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved scores to: {OUTPUT_CSV}")

    topic_model.save(str(MODEL_DIR))
    print(f"Saved BERTopic model to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
