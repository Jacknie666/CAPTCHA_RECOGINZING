🤖 手写数字验证码识别器 | MNIST CNN 模型  Handwritten Digit CAPTCHA Recognizer 🔮

一个使用 TensorFlow/Keras 和 OpenCV 构建的 Python 项目，用于识别简单的手写数字验证码图片。
包含模型训练脚本 (`ModelTraining.py`) 和模型调用脚本 (`Model_Identify.py`)。

✨ 主要功能:
- 图像预处理 (灰度、二值化)
- 基于轮廓的字符分割
- 使用预训练的 CNN 模型进行数字预测 (0-9)
- 可视化识别过程

🚀 快速开始:
1. 将验证码图片放入 `custom_images` 文件夹。
2. 修改 `Model_Identify.py` 中的图片路径。
3. 运行 `python Model_Identify.py` 进行识别！

💡 模型文件: `mnist_cnn_best_model.h5`

欢迎 Star ⭐ 和 Fork 🍴！
