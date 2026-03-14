#!/usr/bin/env bash
# 等 Ollama 啟動後，下載 Qwen3.5-27B abliterated（Q3_K_M 約 12GB，RTX 5070 Ti 16GB 安全）
# Q4_K_M ≈ 16.4GB（超過可用 VRAM，不建議）

set -e

echo "等待 Ollama 服務啟動..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "✓ Ollama 已就緒"

MODEL="hf.co/huihui-ai/Qwen3.5-27B-abliterated-GGUF:Q3_K_M"
echo "開始下載：$MODEL"
docker exec ollama-week3 ollama pull "$MODEL"
echo "✓ 模型下載完成"
echo ""
echo "測試執行："
docker exec ollama-week3 ollama run "$MODEL" "用一句話解釋語意搜尋"
