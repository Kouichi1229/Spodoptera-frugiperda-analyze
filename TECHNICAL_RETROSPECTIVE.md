# 秋行軍蟲影像分類：技術經歷回顧

> 本文件以 Repository 中的 Notebook、訓練輸出、模型檔案說明與 Flask 推論程式為依據，整理曾經實作的技術經驗。這是歷史專案回顧，不代表目前模型已重新訓練或通過獨立測試集驗證。

## 1. 專案目標

使用影像分類模型辨識四種農業害蟲：

1. 秋行軍蟲（Spodoptera frugiperda）
2. 斜紋夜盜（Spodoptera litura）
3. 番茄夜蛾（Helicoverpa armigera）
4. 甜菜夜蛾（Spodoptera exigua）

Repository README 將專案描述為以 Keras API 進行台灣害蟲影像分類，原始標題另列出 VGG19、ResNet152V2、InceptionResNetV3、DenseNet201 與 NASNetLarge 等架構名稱。不過，本次已逐段核對的訓練 Notebook 證據主要確認 VGG19 與 InceptionV3。關聯 Flask Repository 的歷史版本另有 DenseNet201 H5 模型檔案，但目前未找到可核對其訓練流程的 Notebook，因此只記錄為曾存在模型檔案，不將它等同於已驗證的完整訓練成果；ResNet152V2、InceptionResNetV3 與 NASNetLarge 也不列為已核實實作。Notebook 使用 Google Colab 與 Google Drive 載入資料，訓練並儲存模型。

## 2. 專案可證明的技術能力

| 能力領域 | 程式證據 | 可合理描述的經驗 |
|---|---|---|
| Python 深度學習開發 | Jupyter Notebook、TensorFlow/Keras 程式 | 曾使用 Python 建立影像分類訓練流程 |
| CNN 預訓練模型 | VGG19、InceptionV3 | 使用 ImageNet 預訓練 CNN 作為特徵抽取骨幹 |
| 遷移學習 | 將預訓練模型層設為 `trainable=False`，接上自訂分類層 | 具備 Transfer Learning 基礎實作經驗 |
| 影像資料前處理 | 影像尺寸調整、像素縮放、ImageDataGenerator | 曾處理影像尺寸與訓練資料增強 |
| 多類別分類 | 四個類別、Softmax 輸出、categorical crossentropy | 曾建立四分類影像分類模型 |
| 模型訓練 | RMSprop、batch generator、epochs、model.fit | 曾執行模型訓練並觀察 loss／accuracy |
| 模型評估 | confusion matrix、錯誤樣本索引及預測信心值 | 曾實作分類結果檢視與錯誤案例分析 |
| 可解釋性視覺化 | TensorFlow GradientTape、Grad-CAM、OpenCV | 曾產生 CNN 關注區域熱感圖 |
| 模型部署介面 | 另一個 Repository 的 Flask 應用載入 H5 模型 | 曾將影像分類模型串接到 Web 推論流程 |

## 3. 訓練資料與模型設計

### 資料集

Notebook 執行輸出顯示，資料載入器讀取到 **589 張影像、4 個類別**。資料以類別資料夾組織，並透過 Keras `flow_from_directory()` 建立批次資料產生器。

資料來源連結記錄在原始 README 中。由於目前無法僅憑 Notebook 判定資料集的完整授權、採集條件、重複影像及各類別的獨立來源，作品集不應對資料來源品質作超出證據的描述。

### VGG19 Notebook

檔案：`VGG19.ipynb`

程式可確認的流程：

1. 從 Google Drive 複製並解壓縮影像資料。
2. 使用 Keras Applications 載入 ImageNet 預訓練 VGG19。
3. 設定輸入影像大小為 224 × 224 × 3，並使用 `include_top=False` 移除原始分類頂層。
4. 將預訓練模型的所有層設為不可訓練，保留其既有特徵表示。
5. 取出 `block5_pool` 輸出，接上 Flatten、1024 個 ReLU 神經元的 Dense 層、Dropout(0.2)，最後接四個 Softmax 輸出。
6. 使用 RMSprop（learning rate 0.0001）、categorical crossentropy 及 accuracy 指標編譯模型。
7. 使用 ImageDataGenerator 做 rescale、旋轉、寬高位移、shear、zoom 及水平翻轉。
8. 使用 batch size 4、每個 epoch 100 steps、訓練 50 epochs，並將模型存成 H5。

這代表你曾實作「凍結預訓練 CNN 特徵抽取層，再訓練自訂分類頭」的遷移學習流程。

### InceptionV3 Notebook

檔案：` InceptionV3.ipynb`（檔名開頭有空白）

程式可確認的流程：

1. 載入 ImageNet 預訓練 InceptionV3，輸入尺寸設定為 150 × 150 × 3，`include_top=False`。
2. 凍結預訓練模型層，取出 `mixed7` 作為特徵輸出。
3. 接上 Flatten、Dense(1024, ReLU)、Dropout(0.2) 及四類 Softmax 分類層。
4. 使用 RMSprop（learning rate 0.0001）與 categorical crossentropy。
5. 使用 ImageDataGenerator 做與 VGG19 相近的影像增強。
6. 使用 batch size 4、100 steps、訓練 100 epochs，並儲存 H5 模型。
7. 後續載入模型，計算分類機率、預測類別及混淆矩陣。

**重要：** VGG19 與 InceptionV3 的輸入尺寸不同，推論端必須與各自模型的訓練前處理保持一致。

## 4. 分類評估與結果解讀

InceptionV3 Notebook 中可以看到：

- 以 `flow_from_directory(..., shuffle=False)` 取得資料及標籤。
- 將 one-hot 標籤及模型預測轉為類別索引。
- 使用 Scikit-learn `confusion_matrix` 建立混淆矩陣。
- 以列正規化方式呈現各真實類別的分類比例。
- 找出預測錯誤樣本，列出原始標籤、預測標籤及模型預測信心值，並顯示錯誤影像。

Notebook 中的混淆矩陣輸出為：

```text
[[153   0   5   5]
 [  1  93   4   6]
 [  0   3 214   4]
 [  0   0   1 100]]
```

這個矩陣的總數為 589，與該資料集影像數相同。

### 評估限制：目前不能當成獨立測試集成績

Notebook 的訓練資料與後續評估流程都指向同一個 `base_dir`。評估時雖然設定 `shuffle=False`，但仍使用 `train_datagen`，而不是獨立的 validation/test generator；Notebook 也沒有顯示將資料分成訓練集與測試集的程式。

因此，這個混淆矩陣只能視為當時在同一資料目錄上執行的分類結果紀錄，不能當成獨立測試集準確率，也不能據此宣稱模型具有良好泛化能力。若要重新評估，應先建立不重疊的 train/validation/test split，並確認資料來源或近似影像沒有跨集合洩漏。

## 5. Grad-CAM 模型關注區域視覺化

檔案：`CAM_simple.ipynb`

Notebook 使用 InceptionV3 與 TensorFlow GradientTape：

1. 取得模型預測類別。
2. 擷取指定卷積層（Notebook 使用 `conv2d_93`，另一段使用 `mixed7`）的輸出。
3. 計算目標類別分數對卷積特徵圖的梯度。
4. 對梯度做 Global Average Pooling，作為各特徵通道的權重。
5. 將權重與特徵圖相乘並加總，產生熱感圖。
6. 將熱感圖正規化、放大至原始影像尺寸，再使用 OpenCV 色彩映射疊加至原圖。

這項工作屬於 CNN 可解釋性視覺化的基礎實作，可用來觀察模型在分類時對影像哪些區域較敏感。

**解讀限制：** Grad-CAM 顯示的是模型對特定類別輸出的梯度加權特徵區域，不等於模型真正理解昆蟲形態，也不能單憑熱感圖證明模型使用了正確的生物學特徵。

## 6. Flask 推論 Web App（關聯 Repository）

關聯 Repository：[Classification_webapp](https://github.com/Kouichi1229/Classification_webapp)

已從該 Repository 的 Git commit 歷史確認 Spodopterafrugiperda_app.py 的 Flask 程式，以及三個 H5 模型檔案曾經提交至版本庫。由於目前透過檔案 API 無法穩定讀取該 Repository 的完整最新檔案樹，以下描述以可讀取的歷史 commit 內容為依據，不推定目前部署狀態。

程式可確認的功能：

- 使用 Flask 建立 Web 應用，提供首頁、VGG19 與 DenseNet201 頁面路由；首頁回傳 InceptionV3 頁面。
- 提供 /predictVGG19、/predictInceptionV3、/predictDenseNet201 三個 POST 推論路由。
- 接收上傳圖片，使用 Werkzeug secure_filename() 處理檔名並儲存暫存檔。
- 使用 TensorFlow Keras load_model() 載入 H5 模型，PIL 調整影像大小，NumPy 組成模型輸入。
- 使用 predict_classes() 取得預測類別，再透過類別字典轉換成四種害蟲名稱。
- 推論完成後呼叫 os.remove() 刪除暫存圖片。
- Procfile 設定以 Gunicorn 啟動 Spodopterafrugiperda_app:app。
- requirements.txt 記錄的歷史環境包含 Flask 1.1.2、TensorFlow GPU 2.5.0 RC、Keras 2.4.3、NumPy 1.19.5、Pillow 8.2.0 與 Gunicorn 20.1.0；這是版本庫中的依賴清單，不代表目前已在新環境成功重建。
- 版本庫歷史曾包含 VGG19_BUG.h5、InceptionV3_bug.h5、DenseNet201_bug.h5 三個模型檔案（以 Git LFS pointer 記錄）。

可追溯的程式版本：
- [Flask app 初始程式 commit](https://github.com/Kouichi1229/Classification_webapp/commit/a44b79616fecfebd2e595430cbd78a65c4663381)
- [模型檔案加入版本庫的 commit](https://github.com/Kouichi1229/Classification_webapp/commit/510ba70a1a3a04637a7b9477fa8d51fbcb899084)
- [調整推論影像尺寸的 commit](https://github.com/Kouichi1229/Classification_webapp/commit/b0321f0525a6733335b5c86682a50453837fdc64)

### 推論端需要留意的歷史程式風險

以下是靜態檢視程式與 commit 差異後發現的問題，不代表已在當年部署環境實際發生：

1. **推論尺寸曾被修改，且與 Notebook 設定存在落差：** 初始 Flask 程式使用 VGG19 30×30、InceptionV3 150×150、DenseNet201 30×30；後續 commit 將三者改為 VGG19 224×224、InceptionV3 229×229、DenseNet201 224×224。已核對的 Notebook 設定則為 VGG19 224×224、InceptionV3 150×150。VGG19 後續尺寸與 Notebook 相符，但 InceptionV3 後續尺寸 229×229 與 Notebook 的 150×150 不同；DenseNet201 尚無已核對的訓練 Notebook 可供比對。這表示推論前處理版本需要再與實際 H5 模型核對，不能直接假設完全一致。
2. **像素縮放可能不一致：** 已核對的 Notebook 訓練資料產生器使用 rescale=1./255.；Flask app 讀取影像、調整尺寸後直接轉為 NumPy array，沒有看到相同的 rescale。若模型確實由該 Notebook 流程訓練，輸入尺度可能不一致。
3. **類別索引可能不一致：** InceptionV3 Notebook 的 class_indices 輸出順序為 Helicoverpa armigera、Spodoptera exigua、Spodoptera frugiperda、Spodoptera litura；Flask app 的類別字典順序為 Spodoptera frugiperda、Spodoptera litura、Helicoverpa armigera、Spodoptera exigua。若 Flask 載入的模型使用前述 Notebook 的類別索引，推論標籤可能錯置；仍須以實際模型訓練時的類別映射確認。
4. **模型每次請求重新載入：** 三個影像處理函式都在函式內呼叫 load_model()，可能造成每次推論都重新載入模型，增加延遲與資源消耗。
5. **使用舊版 Keras API：** 程式使用 predict_classes()；此 API 在較新的 Keras 版本已不再適用，現代實作通常改用 predict() 搭配 argmax。
6. **上傳處理韌性不足：** 暫存檔刪除沒有包在 try/finally，若預測中途失敗，檔案可能殘留；程式也沒有明確將圖片轉為 RGB。
7. **部署安全與驗證：** 程式以 debug=True 啟動 Flask，且未見完整的上傳檔案大小、格式及內容驗證，不宜直接視為已具備生產環境部署安全性。

**能力描述界線：** 可確認曾實作 Flask 圖片上傳、H5 模型載入與多模型推論路由；目前不能據此宣稱三個模型都已完成正確的端到端驗證，亦不能宣稱 Web App 現在仍可正常部署。

## 7. 個人技術經歷的合理定位

### 履歷描述（中文）

使用 TensorFlow／Keras 進行四類農業害蟲影像分類，採用 ImageNet 預訓練 VGG19 與 InceptionV3 建立遷移學習模型，透過 ImageDataGenerator 執行影像增強，並實作模型訓練、分類結果分析、混淆矩陣及 Grad-CAM 熱感圖。另以 Flask 串接 H5 模型，建立圖片上傳與分類結果回傳的 Web 推論介面。

### 面試口頭說明

這個專案是我在 AI 訓練期間做的電腦視覺專案。我們使用四種害蟲影像建立分類資料集，當時我用 Keras 的預訓練 CNN，例如 VGG19 和 InceptionV3，凍結原本的特徵抽取層，再接上自己的分類層進行遷移學習。我也處理了影像增強、模型訓練和混淆矩陣，另外嘗試用 Grad-CAM 看模型主要關注影像的哪些區域。後續還有 Flask Web App，讓使用者上傳圖片並取得分類結果。回頭檢視後，我也發現當時評估資料切分與推論前處理還有需要改善的地方。

## 8. 能力層級判讀

| 技術 | 專案證據 | 建議定位 |
|---|---|---|
| TensorFlow／Keras | 建立及訓練 CNN 分類模型 | 有深度學習實作經驗，需重新熟悉 API |
| Transfer Learning | 凍結預訓練骨幹、建立分類頭 | 具備遷移學習基礎實作 |
| CNN 架構應用 | VGG19、InceptionV3 | 使用過預訓練架構，不代表能自行設計大型 CNN |
| Image Augmentation | ImageDataGenerator 多種增強 | 有影像前處理經驗 |
| 模型評估 | 混淆矩陣、錯誤樣本檢視 | 有基礎評估實作，但獨立測試集驗證不足 |
| Explainable AI | Grad-CAM | 有可解釋性視覺化的實作接觸 |
| Flask | 上傳、路由、模型推論、結果回傳 | 有 Python Web 推論串接經驗 |
| 模型部署 | H5 載入與 Gunicorn Procfile | 有簡易部署配置經驗；不等同維運生產級服務 |

## 9. 本專案對後續 AI 應用開發的價值

這段經驗與目前轉向 AI 應用開發有直接關聯：

- 理解模型不是只有輸入與輸出，還需要資料前處理、模型載入及標籤映射。
- 接觸過模型訓練與推論服務之間的銜接。
- 曾把模型封裝成 Web API／Web 頁面可使用的功能。
- 能辨識模型評估與部署流程中常見的資料一致性問題。
- Grad-CAM 經驗可延伸至 AI 系統的可解釋性概念。

**專案定位：** Python 電腦視覺、CNN 遷移學習、影像資料增強、模型評估與 Grad-CAM 視覺化，並延伸至 Flask 影像分類推論應用。

本文件不宣稱模型已通過獨立測試集驗證，也不宣稱 Flask app 現在仍可正常部署。
