import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import onnx
import json
from sklearn.preprocessing import StandardScaler

# 特徴抽出（Numpyベース）
def extract_features(img, mean=None, std=None):
    # img は PyTorch の Tensor で、形状は (28, 28) を想定
    img_np = img.numpy() # NumPy 配列に変換 (0-1の範囲に正規化されているはず)

    # 入力1: 総ピクセル密度 (Total Pixel Density)
    # img_np は0-1に正規化されているため、そのまま合計を画素数で割る
    total_pixel_value = img_np.sum()
    total_pixel_density = total_pixel_value / (28 * 28) # 0-1 の範囲

    # 入力2: 上半分と下半分のピクセル密度の差 (Top-Bottom Density Difference)
    upper_half = img_np[:14, :].sum()
    lower_half = img_np[14:, :].sum()
    # 各半分のピクセル数を考慮して正規化
    upper_half_density = upper_half / (14 * 28)
    lower_half_density = lower_half / (14 * 28)
    top_bottom_density_diff = upper_half_density - lower_half_density
    # 差の範囲は-1から1。そのまま使用。

    # 入力3: 垂直方向の重心 (Vertical Centroid)
    # ピクセル値が0の場合、分母が0になることを避けるため、1e-6 を加える
    sum_of_pixel_values = img_np.sum()
    if sum_of_pixel_values < 1e-6: # 画像が完全に真っ白な場合など、0除算を避ける
        vertical_centroid = 0.5 # 中央に設定
    else:
        # 各行のY座標（0-27）を重み付けして合計し、総ピクセル値で割る
        y_coords = np.arange(28).reshape(-1, 1) # (28, 1)
        weighted_sum_y = (y_coords * img_np).sum()
        vertical_centroid = weighted_sum_y / sum_of_pixel_values
        vertical_centroid = vertical_centroid / 27.0 # 0-27の範囲を0-1に正規化

    # 入力4: 水平方向の交差数 (Horizontal Crossing Count)
    crossing_count = 0
    # 各行を走査
    for row in range(28):
        current_row = img_np[row, :]
        # 閾値を設定（ピクセルが描画されているとみなす閾値、例: 0.1）
        threshold = 0.05 # MNISTはグレースケールなので、完全に0でなくても少し値がある場合がある
        is_drawing = current_row > threshold
        # 0から1、または1から0への変化をカウント
        for i in range(27):
            if is_drawing[i] != is_drawing[i+1]:
                crossing_count += 1
    # 最大交差数で正規化 (28行 * 2 (開始と終了) = 56 が考えられるが、
    # 数字の形状を考慮すると行あたり2-4回程度が多いので、経験的に調整)
    # 最大交差数は、例えば全ての行で2回の交差があるとして 28 * 2 = 56
    # もしくは、もっと複雑な数字で最大でどれくらいありえるかを考慮し、大きめに設定
    # 経験的に、MNISTの数字では1行あたり4回程度の交差が最大と仮定して、28*4=112で正規化してみる。
    max_possible_crossings = 112 # 28行 * 各行で最大4回の交差
    horizontal_crossing_count = crossing_count / max_possible_crossings

    features = np.array([
        total_pixel_density,
        top_bottom_density_diff,
        vertical_centroid,
        horizontal_crossing_count,
    ], dtype=np.float32)

    if mean is not None and std is not None:
        features = (features - mean) / (std + 1e-6)

    return features

# ラベル変換関数
def group_label(y):
    if y in [0, 8, 9]:
        return 0
    elif y in [1, 3, 5, 7]:
        return 1
    else:
        return 2

# カスタムデータセット：MNISTから特徴抽出
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset):
        self.data = []
        self.labels = []
        for img, label in mnist_dataset:
            feat = extract_features(img)
            self.data.append(feat)
            self.labels.append(group_label(label))
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32) # 特徴量はfloat32
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ネットワーク定義
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 9)
        self.fc2 = nn.Linear(9, 11)
        self.fc3 = nn.Linear(11, 6)
        self.fc4 = nn.Linear(6, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 各層の出力を変数に保持
        out_fc1 = self.fc1(x)
        out_fc1_relu = self.relu(out_fc1) # out_fc1_reluをONNX出力として定義したい場合

        out_fc2 = self.fc2(out_fc1_relu)
        out_fc2_relu = self.relu(out_fc2) # out_fc2_reluをONNX出力として定義したい場合

        out_fc3 = self.fc3(out_fc2_relu)
        out_fc3_relu = self.relu(out_fc3) # out_fc3_reluをONNX出力として定義したい場合

        output = self.fc4(out_fc3_relu)
        
        # ONNXエクスポートの出力として、最終出力と中間層のReLU出力をすべて返す
        # ONNX.jsでこれらの名前でアクセス可能になる
        return output, out_fc1_relu, out_fc2_relu, out_fc3_relu

# メイン処理
def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0))  # 1x28x28 -> 28x28
    ])

    mnist_train = datasets.MNIST(root="./data", train=True, download=False, transform=transform)

    dataset = FeatureDataset(mnist_train)

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 学習
    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # モデルの出力を取得
            outputs, _, _, _ = model(inputs) 

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / train_size
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # 評価ループも同様に修正
    model.eval()
    correct = 0
    total = 0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs, _, _, _ = model(inputs) 

            _, preds = torch.max(outputs, 1)
            preds_all.append(preds.cpu())
            labels_all.append(labels.cpu())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)

    # 各クラスのprecision計算
    for cls in range(3):
        tp = ((preds_all == cls) & (labels_all == cls)).sum().item()
        fp = ((preds_all == cls) & (labels_all != cls)).sum().item()
        precision = tp / (tp + fp + 1e-6)
        print(f"Class {cls} Precision: {precision:.4f}")

    # mAP（平均precision）
    precisions = []
    for cls in range(3):
        tp = ((preds_all == cls) & (labels_all == cls)).sum().item()
        fp = ((preds_all == cls) & (labels_all != cls)).sum().item()
        precisions.append(tp / (tp + fp + 1e-6))
    map_score = np.mean(precisions)
    print(f"mAP (mean average precision): {map_score:.4f}")

    

    # モデルをONNX形式で保存
    model_dir = "./model_for_web"
    os.makedirs(model_dir, exist_ok=True)
    onnx_path = os.path.join(model_dir, "feature_classifier.onnx")

     # ダミー入力を準備 (4つの特徴量)
    dummy_input = torch.randn(1, 4).to(device) 

    # ONNX形式でエクスポート
    # output_names を、forwardメソッドで返される順序に合わせて指定
    torch.onnx.export(model, 
                      dummy_input, 
                      onnx_path, 
                      export_params=True, 
                      opset_version=10, 
                      do_constant_folding=True, 
                      input_names=['input'], 
                      output_names=['output_final', 'output_layer1_relu', 'output_layer2_relu', 'output_layer3_relu'], # わかりやすい名前
                      dynamic_axes={'input' : {0 : 'batch_size'},    # 動的バッチサイズ
                                    'output_final' : {0 : 'batch_size'},
                                    'output_layer1_relu' : {0 : 'batch_size'},
                                    'output_layer2_relu' : {0 : 'batch_size'},
                                    'output_layer3_relu' : {0 : 'batch_size'}})
    print(f"モデルがONNX形式で '{onnx_path}' に保存されました。")


if __name__ == "__main__":
    main()
