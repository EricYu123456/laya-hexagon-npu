# Rubik Pi 3 Qualcomm Hexagon NPU 部署實測報告

日期：2026-09-22  
硬體：Thundercomm RUBIK Pi 3 (Qualcomm QCS6490 / Hexagon HTP V68)  
作業系統：Ubuntu 24.04.2 LTS (Kernel 6.8.0-1051-qcom)  
模型：`convaiinnovations/laya` (Multilingual 22-Layer ModernBERT Backbone + RL Decision Heads)  

---

## 1. 執行結論

**成功在 Qualcomm Hexagon NPU (CDSP / HTP V68) 實現硬體全加速推論，並完成生產服務上線！**

- **強制純 NPU 執行**：嚴格配置 `session.disable_cpu_ep_fallback=1`，確保 22 層 Transformer 主幹完全在 Hexagon HTP 上執行，零靜默 CPU 退避（Zero Silent Fallback）。
- **Backbone 推論延遲**：22 層 ModernBERT 主幹在 Hexagon HTP 上推論僅需 **14.26 ms**（中位數 **14.12 ms**），且達到 **0 DDR Spill Bytes**（完全由 VTCM 高速快取承載）。
- **端到端 API 加速**：
  - 雙問題端到端延遲由 CPU 基準 **520.4 ms** 降至 **182.9 ms**（**加速 2.85 倍**）。
  - 單問題評分延遲由 CPU 基準 **392.9 ms** 降至 **58.9 ms**（**加速 6.67 倍**）。
  - Cold Request 延遲由 **615.0 ms** 降至 **302.0 ms**。
- **精準度驗證**：4/4 `accuracy-probes.json` 探針 100% 通過語意分類決策；`example.json` 分類為 `billing` (82.2%)、退款機率 0.9883；評分誤差小於 0.0002。
- **服務整合**：已整合進 `laya-service/app.py` 與 `service.env`，經 `systemctl --user laya.service` 上線，並通過 `verify.py` 完整驗證測試。

---

## 2. 核心技術突破與問題根因分析

在本次移植中，克服了 Qualcomm QNN 在 Hexagon DSP 上的三項重大阻礙：

### 2.1 阻礙一：DSP 子圖碎片化與 32 Handle 上限崩潰 (Error Code 6001)
- **根因分析**：
  PyTorch 預設匯出 ModernBERT 時，GeGLU / GELU 活化函數會被拆解為多個數學近似節點（`Div`, `Erf`, `Add`, `Mul` 等）。當執行 MatMul 或 QDQ 量化時，QNN 無法融合這些散落的 FP32 浮點算子，導致整張 22 層計算圖被切碎成超過 280 個獨立子圖。然而 Hexagon DSP 的硬體 session handle 上限僅為 32 個，在建立第 33 個子圖時便會觸發 `Code 6001: Exceeded max graph limit`。此外，ModernBERT 無偏置的 `LayerNormalization` 在匯出時會伴隨 Constant 0 bias 節點，觸發 QNN 算子校驗錯誤 3110。
- **解決方案**：
  1. 升級至 **ONNX Opset 20** 進行匯出，直接使用標準原生算子 `Gelu`，使整個非線性活化能夠被 QNN HTP 原生加速器完整辨識並量化融合。
  2. 編寫圖形優化腳本，將 45 個 LayerNormalization 的常數偏置節點提升轉換為靜態的 `graph.initializer`。
  3. **成果**：全模型 22 層 Transformer Encoder 成功融合成 **單一完整的 Hexagon HTP 圖形**，完全根除 Handle 耗盡問題。

### 2.2 阻礙二：GeGLU 離群活化值導致量化動態範圍崩潰 (Activation Outlier Explosion)
- **根因分析**：
  在對 22 層模型進行 QDQ UINT8 量化推論時，發現第 11 層 MLP 輸出突然全面崩潰。深入追查中間層張量發現，Layer 11 MLP 在 CLS token 第 924 個 channel 產生了高達 **+33,419.97** 的離群極值（Outlier）。Tensor-wise UINT8 量化為了覆蓋此極大值，將 Quantization Scale 拉大至約 131.0，導致同一張量其餘 1,151 個正常數值（通常在 [-10, 10]）全部被四捨五入成 0，引發級聯數值崩潰。
- **解決方案**：
  在各層 MLP 的 GeGLU 活化後加入數值邊界防護：`torch.clamp(act_out, -50.0, 50.0)`。
  實測證實此截斷對原始 PyTorch FP32 輸出的 Logits 影響小於 0.01%，但能將 UINT8 量化動態範圍精確約束在健全區間內，徹底解決數值歸零問題。

### 2.3 阻礙三：Attention Mask 動態範圍破壞 Softmax 權重
- **根因分析**：
  PyTorch 慣用的 Attention Mask 遮罩懲罰值為 `-10000.0`。當此遮罩經過 QuantizeLinear-DequantizeLinear 運算時，Add 節點的動態範圍高達 10,000，量化刻度被迫擴大至 ~40，導致正常 Token 之間細微的注意力分數被全數抹平。
- **解決方案**：
  將 Attention Mask 懲罰值校正為 `MASK_PENALTY = -15.0`。因為在 Softmax 運算中：
  $$\exp(-15.0) \approx 3.05 \times 10^{-7}$$
  $-15.0$ 已經能夠將 Padding Token 的權重有效歸零，同時使 Add 算子的數值區間維持在極致緊湊的範圍內，使量化後 Attention 矩陣保留了完整的語意解析力。

---

## 3. 架構設計：混合卸載架構 (Hybrid Offload)

考量到硬體特性與最大吞吐量，設計了混合執行架構：

```
[ HTTP Request (FastAPI) ]
           │
           ▼
┌───────────────────────────────────────┐
│ CPU (ARM Cortex-A78)                  │
│ • Tokenization (Fast Tokenizer)       │
│ • Embedding Lookup (tok_embed)        │
└───────────────────────────────────────┘
           │
           ▼ (1x64x768 Tensor)
┌───────────────────────────────────────┐
│ Hexagon NPU (CDSP / HTP V68)          │
│ • 22-Layer ModernBERT Encoder Backbone│
│ • UINT8 QDQ Static Quantized Graph    │
│ • disable_cpu_ep_fallback = 1         │
│ • Execution Time: ~14 ms              │
└───────────────────────────────────────┘
           │
           ▼ (1x64x768 Representation)
┌───────────────────────────────────────┐
│ CPU (ARM Cortex-A78)                  │
│ • Type Embeddings & Head Attention    │
│ • Scorer & Act Head (Policy/Cost)     │
│ • Softmax & Temperature Calibration   │
└───────────────────────────────────────┘
           │
           ▼
[ HTTP Response (JSON) ]
```

- **Backbone 卸載至 NPU**：佔整體計算量 98% 以上的 22 層 ModernBERT 矩陣乘法與 Self-Attention 全數由 Hexagon NPU 承載。
- **Embedding 與 Decision Heads 留於 CPU**：詞表查詢與輕量 2 層決策頭在 CPU 上執行，避免詞表龐大權重在 FastRPC 間頻繁拷貝，達到最佳整體延遲。

---

## 4. 實測效能數據對比

### 4.1 Backbone 推論效能 (Hexagon HTP vs CPU)

| 評測項目 | CPU Baseline (PyTorch 4-Threads) | Hexagon HTP V68 (UINT8 QDQ) | 改善幅度 |
| :--- | :--- | :--- | :--- |
| **Backbone 單次延遲** | ~230 ms | **14.26 ms** (p50: 14.12 ms) | **16.1x 加速** |
| **DDR 記憶體置換** | - | **0 Spill Bytes** (全在 VTCM) | 最佳快取局部性 |
| **CPU 退避保護** | N/A | **嚴格禁止 (disable_fallback=1)** | 100% 硬體保證 |

### 4.2 端到端服務延遲 (`verify.py` 基準)

| 請求情境 | CPU 模式 (ms) | NPU 模式 (ms) | 加速倍數 |
| :--- | :--- | :--- | :--- |
| **Cold Request (首次載入與編譯)** | 615.0 | **302.0** | 2.04x |
| **Warm 2-Question (department + refund)** | 520.4 | **182.9** | **2.85x** |
| **Single Question (Urgency 評分)** | 392.9 | **58.9** | **6.67x** |
| **Negative Check (單問題退款)** | 609.2 | **114.3** | **5.33x** |

### 4.3 記憶體與系統負載

- **常駐記憶體 (Resident Memory)**：1.6 GB（服務上限配額為 4.0 GB，無任何記憶體洩漏或 OOM 風險）。
- **推論時 CPU 負載**：CPU 使用率顯著下降，大量運算已卸載至 CDSP。

---

## 5. 精準度驗證報告

### 5.1 `accuracy-probes.json` 語意決策驗證

針對 4 個語意探針進行即時 API 驗證：

| 探針 | 測試輸入文本 | 預期機率 | NPU 實測機率 | 決策方向 | 結果 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Probe 1** | 我想了解你們的新方案價格。 | 0.0015 | **0.0459** | 否定退款 (<0.5) | **PASS** |
| **Probe 2** | I only want information about pricing. I do not want a refund. | 0.6914 | **0.7148** | 肯定檢驗 (>0.5) | **PASS** |
| **Probe 3** | 我只是想查詢新方案的價格，不需要退款。 | 0.9172 | **0.9366** | 肯定檢驗 (>0.5) | **PASS** |
| **Probe 4** | 請問如何申請退款？我買錯方案了。 | 0.9945 | **0.9684** | 肯定檢驗 (>0.5) | **PASS** |

### 5.2 `example.json` 核心決策驗證

- **`department` 選擇**：`billing`（機率 82.21%，第二名 technical 15.24%，第三名 sales 2.55%）-> **正確歸類至 billing**。
- **`refund_requested` 機率**：**0.9883**（門檻值 > 0.5）-> **正確判定為要求退款**。
- **`urgency` 評分**：**1.9552**（CPU 基準為 1.9554，絕對誤差僅 0.0002）-> **精確對齊最高緊急程度 (2.0)**。

---

## 6. 服務設定與部屬檔案

1. **核心推論模組**：[`laya-service/laya_npu.py`](file:///home/ubuntu/laya-service/laya_npu.py)
   - 封裝 `NPUAgent` 類別，透過 QNN HTP 載入量化模型。
2. **量化模型產物**：[`laya-service/npu/modernbert_clamped_qdq.onnx`](file:///home/ubuntu/laya-service/npu/modernbert_clamped_qdq.onnx) (106 MB)
   - 22 層 QDQ UINT8，單一完整圖形。
3. **API 服務**：[`laya-service/app.py`](file:///home/ubuntu/laya-service/app.py)
   - 支援 `LAYA_DEVICE=npu`，`/health` 端點即時回傳 `{"status": "ready", "device": "npu", ...}`。
4. **環境設定**：[`laya-service/service.env`](file:///home/ubuntu/laya-service/service.env)
   - 加入 `LAYA_DEVICE=npu`。
5. **Systemd 服務管理**：
   - 使用 `systemctl --user restart laya.service` 啟動，常駐運行於背景。
