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


from sentence_transformers import SentenceTransformer
model = SentenceTransformer("/root/project/models/text2vec-base-chinese")
emb = model.encode(["政府数据开放平台"], show_progress_bar=False)
print("Embedding shape:", emb.shape)