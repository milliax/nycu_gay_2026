from sentence_transformers import SentenceTransformer
import numpy as np

# ── 文章資料 ──────────────────────────────────────────────────
documents = [
    "如何透過重量訓練增加肌肉量",
    "減脂期間的飲食控制技巧",
    "Python 資料分析入門指南",
    "如何改善睡眠品質",
    "提升心肺耐力的運動方式",
    "大型語言模型的工作原理",
    "高蛋白飲食對健身的幫助",
    "時間管理的五個技巧",
]

# ── 載入模型 ──────────────────────────────────────────────────
model = SentenceTransformer("intfloat/multilingual-e5-small")

# ── 步驟 1：將 documents 轉換成向量（加 passage: 前綴）────────
doc_texts = ["passage: " + doc for doc in documents]
doc_vectors = model.encode(doc_texts, normalize_embeddings=True)  # shape: (8, 384)

# ── 步驟 2：將 query 轉換成向量（加 query: 前綴）─────────────
query = "如何提高肌肉量？"
query_vector = model.encode("query: " + query, normalize_embeddings=True)  # shape: (384,)

# ── 步驟 3：計算 cosine similarity ────────────────────────────
# 因為向量已 normalize（unit vector），dot product 即等於 cosine similarity
similarities = np.dot(doc_vectors, query_vector)  # shape: (8,)

# ── 步驟 4：排序，找出最相近的文章 ───────────────────────────
ranked_indices = np.argsort(similarities)[::-1]  # 由高到低排序

# ── 步驟 5：印出搜尋結果 ──────────────────────────────────────
print(f"🔍 查詢：{query}")
print("=" * 50)
for rank, idx in enumerate(ranked_indices, 1):
    print(f"#{rank}  相似度: {similarities[idx]:.4f}  {documents[idx]}")
