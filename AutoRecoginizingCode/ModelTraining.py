import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split # 用于分割训练集和验证集

# --- 常量定义 ---
IMG_ROWS, IMG_COLS = 28, 28
NUM_CLASSES = 10
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
MODEL_SAVE_PATH = "mnist_cnn_best_model.h5" # 模型保存路径

# (可选) 设置随机种子，以便结果可复现
# np.random.seed(42)
# tf.random.set_seed(42)

# --- 辅助函数 (仅保留训练相关的) ---
def plot_images(images, labels, class_names=None, num_to_plot=10):
    """简单绘制几张图片及其标签。"""
    fig = plt.gcf()
    fig.set_size_inches(12, 5)
    if num_to_plot > 25: num_to_plot = 25
    for i in range(num_to_plot):
        ax = plt.subplot(2, 5, 1 + i) if num_to_plot == 10 else plt.subplot(5, 5, 1+i)
        ax.imshow(images[i].reshape(IMG_ROWS, IMG_COLS), cmap='binary')
        title = f"Label: {labels[i]}"
        if class_names and labels[i] < len(class_names):
            title = f"Label: {class_names[labels[i]]}"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def show_train_history(history, train_metric, val_metric):
    """绘制训练过程中的指标变化曲线。"""
    plt.figure() # 创建新的 figure 以避免重叠
    plt.plot(history.history[train_metric])
    plt.plot(history.history[val_metric])
    plt.title('Train History')
    plt.ylabel(train_metric.capitalize())
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)
    plt.show()


def preprocess_image_array(img_array):
    """预处理图像数组 (reshape, normalize)。"""
    img_array = img_array.reshape(img_array.shape[0], IMG_ROWS, IMG_COLS, 1)
    img_array = img_array.astype('float32') / 255
    return img_array


def load_and_preprocess_data():
    """加载并预处理MNIST数据集。"""
    (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = mnist.load_data() # 保留原始测试集用于可能的后续评估
    print('Original x_Train shape:', x_train_orig.shape)

    x_train_processed = preprocess_image_array(x_train_orig)
    x_test_processed = preprocess_image_array(x_test_orig) # 预处理测试集，以备评估用
    print('Processed x_train shape:', x_train_processed.shape)

    y_train_onehot = to_categorical(y_train_orig, NUM_CLASSES)
    y_test_onehot = to_categorical(y_test_orig, NUM_CLASSES) # 预处理测试集标签
    print('Original y_train[0]:', y_train_orig[0])
    print('One-hot encoded y_train[0]:', y_train_onehot[0])

    return (x_train_processed, y_train_onehot), (x_test_processed, y_test_onehot), \
           (x_train_orig, y_train_orig) # 返回原始训练数据用于可视化


def build_cnn_model():
    """构建CNN模型。"""
    model = Sequential(name="MNIST_CNN_Model")
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print("\nModel Summary:")
    model.summary()
    return model


def train_model_and_save(model, x_train_data, y_train_data, x_val_data, y_val_data, epochs=20, batch_size=128):
    """训练模型，并使用EarlyStopping和ModelCheckpoint保存最佳模型。"""
    print("\nStarting model training...")

    # 回调函数
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
    # ModelCheckpoint 会保存验证集上准确率最高的模型
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')

    history = model.fit(x_train_data, y_train_data,
                        validation_data=(x_val_data, y_val_data),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    print(f"Model training finished. Best model potentially saved to {MODEL_SAVE_PATH}")
    return history


# --- 主程序 ---
if __name__ == "__main__":
    # 0. 准备类别名称（可选，用于可视化训练数据）
    class_names = [str(i) for i in range(NUM_CLASSES)]

    # 1. 加载和预处理数据
    (x_train_processed, y_train_onehot), (x_test_processed, y_test_onehot), \
    (x_train_orig, y_train_orig) = load_and_preprocess_data()

    # 可视化一些预处理前的训练图像，确保数据加载正确
    print("\nDisplaying a few original training images...")
    plot_images(x_train_orig, y_train_orig, class_names=class_names, num_to_plot=10)

    # 2. 从训练集中划分出一部分作为验证集 (例如10% 或 20%)
    x_train_fit, x_val_fit, y_train_fit, y_val_fit = train_test_split(
        x_train_processed, y_train_onehot, test_size=0.1, random_state=42, stratify=y_train_onehot
    ) # stratify 保证训练集和验证集类别分布相似
    print(f"\nTraining data shape: {x_train_fit.shape}")
    print(f"Validation data shape: {x_val_fit.shape}")

    # 3. 构建模型
    cnn_model = build_cnn_model()

    # 4. 训练模型并保存
    # 你可以调整 epochs 和 batch_size
    train_history = train_model_and_save(cnn_model, x_train_fit, y_train_fit, x_val_fit, y_val_fit,
                                         epochs=20, batch_size=64)

    # 5. 可视化训练历史
    print("\nDisplaying training history...")
    show_train_history(train_history, 'accuracy', 'val_accuracy')
    show_train_history(train_history, 'loss', 'val_loss')

    # 6. (可选) 加载保存的最佳模型并评估其在测试集上的表现
    #    这一步是为了确认模型保存和加载正常，以及了解模型的泛化能力
    #    如果你完全不需要这一步，可以注释掉或删除
    print(f"\nLoading best model from {MODEL_SAVE_PATH} for optional final evaluation...")
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            best_model = load_model(MODEL_SAVE_PATH)
            print("Successfully loaded the best model.")
            print("\nEvaluating the loaded best model on the test set (optional step):")
            scores = best_model.evaluate(x_test_processed, y_test_onehot, verbose=1)
            print(f'Test loss on loaded best model: {scores[0]:.4f}')
            print(f'Test accuracy on loaded best model: {scores[1]:.4f}')
        except Exception as e:
            print(f"Could not load or evaluate the best model from {MODEL_SAVE_PATH}: {e}")
            print("If training just completed, the model in memory (cnn_model) might be the one with 'restored best weights' if EarlyStopping triggered.")
            print("Evaluating current model in memory on test set:")
            scores = cnn_model.evaluate(x_test_processed, y_test_onehot, verbose=1)
            print(f'Test loss on current model: {scores[0]:.4f}')
            print(f'Test accuracy on current model: {scores[1]:.4f}')
    else:
        print(f"Model file {MODEL_SAVE_PATH} not found. This might happen if training was too short or no improvement was seen by ModelCheckpoint.")
        print("Evaluating current model in memory on test set:")
        scores = cnn_model.evaluate(x_test_processed, y_test_onehot, verbose=1)
        print(f'Test loss on current model: {scores[0]:.4f}')
        print(f'Test accuracy on current model: {scores[1]:.4f}')


    print("\n--- Model Training Script Finished ---")