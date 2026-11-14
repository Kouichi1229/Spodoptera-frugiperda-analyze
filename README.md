# 台灣農業害蟲影像分類｜Spodoptera frugiperda analyze

本專案使用 Python、TensorFlow／Keras 與預訓練卷積神經網路（CNN），進行四類農業害蟲影像分類，並包含模型訓練、分類結果檢視及 Grad-CAM 熱感圖視覺化。

## 專案內容

- **分類類別：** 秋行軍蟲（Spodoptera frugiperda）、斜紋夜盜（Spodoptera litura）、番茄夜蛾（Helicoverpa armigera）、甜菜夜蛾（Spodoptera exigua）。
- **已核對的模型實作：** VGG19、InceptionV3 遷移學習。
- **主要工具：** Python、TensorFlow／Keras、Jupyter Notebook、Google Colab、OpenCV、Scikit-learn。
- **影像處理：** 尺寸調整、像素縮放及 ImageDataGenerator 資料增強。
- **模型分析：** 混淆矩陣、錯誤樣本檢視、Grad-CAM。
- **延伸應用：** 關聯的 Flask Web App 可接收圖片並呼叫 H5 模型進行推論，詳見技術回顧文件。

> 原始 README 曾列出 ResNet152V2、InceptionResNetV3、DenseNet201、NASNetLarge 等模型名稱；目前技術回顧僅將已能從 Notebook 程式確認的 VGG19 與 InceptionV3 列為已核實實作，其他架構不先視為已完成。

## 專案素材

- [資料集](https://reurl.cc/Oq3AdD)
- [模型權重](https://reurl.cc/5qWQ3R)
- [專案簡報（PPT）](https://docs.google.com/presentation/d/1N8tuPnfiaZLO6CAI4XkS3bpuOgPriH0d/edit?usp=sharing&ouid=102938102531618732336&rtpof=true&sd=true)

## 執行成果與視覺化

![模型訓練結果](https://github.com/Kouichi1229/-Spodoptera-frugiperda-analyze/blob/main/%E9%82%B1%E8%A1%8C%E8%BB%8D%E9%87%8D%E6%A8%A1%E5%9E%8B%E8%B7%91%E5%88%86.png?raw=true)

![Grad-CAM 熱感圖](https://github.com/Kouichi1229/-Spodoptera-frugiperda-analyze/blob/main/IV3VGG19%E7%86%B1%E6%84%9F%E5%9C%96.png?raw=true)

![特徵視覺化](https://github.com/Kouichi1229/-Spodoptera-frugiperda-analyze/blob/main/%E7%89%B9%E5%BE%B5%E8%A1%A8%E7%A4%BA.png?raw=true)

## 技術經歷回顧

模型架構、訓練流程、資料增強、評估限制、Grad-CAM、Flask 推論應用及可用於履歷／面試的經歷描述，請參閱：

[TECHNICAL_RETROSPECTIVE.md](TECHNICAL_RETROSPECTIVE.md)

## 評估注意事項

Notebook 中保存的混淆矩陣來自當時的資料載入與評估流程；目前未找到足以證明使用獨立測試集的資料切分證據，因此不將該結果宣稱為獨立測試集準確率。完整限制與程式檢視發現請見技術回顧文件。
