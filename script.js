// ====== Canvas描画関連の初期設定 ======
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const predictButton = document.getElementById('predictButton');
const clearButton = document.getElementById('clearButton');
const predictionResultSpan = document.getElementById('predictionResult');
const probSpans = {
    0: document.getElementById('prob0'),
    1: document.getElementById('prob1'),
    2: document.getElementById('prob2')
};

let isDrawing = false;
const PEN_WIDTH = 20; // 描画する線の太さ

// キャンバスをクリアする関数
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'black'; // 背景色を黒に設定
    ctx.fillRect(0, 0, canvas.width, canvas.height); // 背景を黒で塗りつぶす
    predictionResultSpan.textContent = '---';
    probSpans[0].textContent = '---';
    probSpans[1].textContent = '---';
    probSpans[2].textContent = '---';
}

// 初期描画
clearCanvas();
ctx.lineWidth = PEN_WIDTH;
ctx.lineCap = 'round';
ctx.strokeStyle = 'white'; // 描画色を白に

// マウスイベントリスナー
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});

canvas.addEventListener('mouseup', () => {
    isDrawing = false;
    ctx.closePath();
});

canvas.addEventListener('mouseleave', () => {
    isDrawing = false;
    ctx.closePath();
});

clearButton.addEventListener('click', clearCanvas);

// ====== ONNXモデルと正規化パラメータの読み込み ======
let session;

async function loadModel() {
    try {
        session = new onnx.InferenceSession();
        // モデルファイルのパスは、HTMLファイルからの相対パス
        await session.loadModel('./model_for_web/feature_classifier.onnx');
        console.log('ONNXモデルを読み込みました。');



        predictButton.disabled = false; // モデルが読み込めたらボタンを有効化
    } catch (error) {
        console.error('モデルまたはパラメータの読み込みに失敗しました:', error);
        predictButton.disabled = true;
        predictionResultSpan.textContent = 'エラー: モデル読み込み失敗';
    }
}

// ページロード時にモデルを読み込む
window.addEventListener('load', loadModel);


// ====== NN構造図の描画とノード発火の可視化 ======
const nnSvg = document.getElementById('nnSvg');
const svgWidth = 500;
const svgHeight = 400;

// モデル構造の定義 (ノード数と位置)
const layerConfig = [
    { name: "Input", nodes: 4, x: 50 },
    { name: "Hidden1", nodes: 9, x: 150 },
    { name: "Hidden2", nodes: 11, x: 250 },
    { name: "Hidden3", nodes: 6, x: 350 },
    { name: "Output", nodes: 3, x: 450 }
];

const nodeRadius = 8;
const activeScale = 1.5; // 発火時のノードの半径拡大率

function drawNN() {
    nnSvg.innerHTML = ''; // SVGをクリア
    // 💡 修正点: svg要素のheight属性も更新
    nnSvg.setAttribute('height', svgHeight);

    let prevLayerNodes = null;

    layerConfig.forEach((layer, layerIdx) => {
        // 💡 修正点: ノードY座標の計算ロジックをより汎用的に調整
        // 各層の中心を基準に、ノードを上下に配置
        const totalNodeHeight = layer.nodes * (nodeRadius * 2);
        const totalSpacingHeight = svgHeight - totalNodeHeight;
        const nodeSpacing = totalSpacingHeight / (layer.nodes + 1); // ノード間の均等なスペース

        const nodes = [];
        for (let i = 0; i < layer.nodes; i++) {
            // 各ノードのY座標は、上端からのオフセット + (ノード間のスペース + ノードの直径) * 現在のノードインデックス
            const nodeY = nodeSpacing + nodeRadius + i * (nodeRadius * 2 + nodeSpacing);
            
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', layer.x);
            circle.setAttribute('cy', nodeY);
            circle.setAttribute('r', nodeRadius);
            circle.setAttribute('class', `node ${layer.name.toLowerCase()}-node`);
            circle.setAttribute('id', `node-L${layerIdx}-N${i}`); // 後でアクセスするためのID
            nnSvg.appendChild(circle);
            nodes.push({ x: layer.x, y: nodeY, element: circle });

            // ノード番号のテキスト (入力層と出力層は番号、隠れ層は空でいいかも)
            // 💡 修正点: テキストのY座標もノードYに合わせる
            if (layerIdx === 0 || layerIdx === layerConfig.length - 1) { // 入力層または出力層
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', layer.x);
                text.setAttribute('y', nodeY + nodeRadius + 10); // ノードの下に配置
                text.setAttribute('text-anchor', 'middle');
                text.setAttribute('font-size', '8px');
                text.setAttribute('fill', '#555');
                text.textContent = i;
                nnSvg.appendChild(text);
            }
        }

        // 前の層との接続線を描画
        if (prevLayerNodes) {
            prevLayerNodes.forEach(prevNode => {
                nodes.forEach(currNode => {
                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', prevNode.x + nodeRadius); // 前のノードの右端から
                    line.setAttribute('y1', prevNode.y);
                    line.setAttribute('x2', currNode.x - nodeRadius); // 現在のノードの左端まで
                    line.setAttribute('y2', currNode.y);
                    line.setAttribute('stroke', '#ccc'); // デフォルトの線の色
                    line.setAttribute('stroke-width', 0.5);
                    nnSvg.insertBefore(line, nnSvg.firstChild); // ノードの下に線を配置
                });
            });
        }
        prevLayerNodes = nodes;
    });
}

// ノードの発火状態を更新する関数
// 💡 修正: 引数の順番を、呼び出し側 (predictButton.addEventListener) に合わせる
function updateNNVisualization(features, intermediateOutputs, finalOutputProbabilities) { 
    // すべてのノードをリセット
    nnSvg.querySelectorAll('.node').forEach(nodeElement => {
        // レイヤーごとのデフォルト色に戻す
        if (nodeElement.classList.contains('input-node')) {
            nodeElement.style.fill = '#a0c4ff'; 
        } else if (nodeElement.classList.contains('output-node')) {
            nodeElement.style.fill = '#ffadad';
        } else {
            nodeElement.style.fill = 'lightgray'; // 隠れ層のデフォルト色
        }
        nodeElement.setAttribute('r', nodeRadius); // デフォルトの半径に戻す
    });

    // 💡 修正: 入力層 (Features) の可視化ロジックを追加
    // 入力ノードは特徴量の絶対値に基づいて色を変化させる
    for (let i = 0; i < features.length; i++) {
        const nodeElement = document.getElementById(`node-L0-N${i}`); // Layer 0 (Input)
        if (nodeElement) {
            const featureValue = Math.abs(features[i]); // 絶対値を取る (正規化で負の値もありえるため)
            // 入力特徴量のスケールは -2 から 2 程度を想定。0-1に正規化して色強度を調整。
            // ここでの 0.5 は、特徴量がある程度大きな値（例: 2.0）でも色に飽和が起きにくいように調整しています。
            const colorIntensity = Math.min(1, Math.max(0, featureValue * 0.5)); 
            nodeElement.style.fill = `rgb(${255 * colorIntensity}, ${255 * colorIntensity}, ${255 * colorIntensity})`; // 白に近づく
            nodeElement.setAttribute('r', nodeRadius * (1 + colorIntensity * 0.5)); // 活性化に応じて半径を少し大きく
        }
    }

    // 隠れ層1 (output_layer1_relu)
    const layer1Values = intermediateOutputs.output_layer1_relu.data;
    for (let i = 0; i < layer1Values.length; i++) {
        const nodeElement = document.getElementById(`node-L1-N${i}`); // Layer 1 (Hidden1)
        if (nodeElement) {
            const activation = layer1Values[i];
            const colorIntensity = Math.min(1, Math.max(0, activation)); // 0-1にクリップ
            nodeElement.style.fill = `rgb(${colorIntensity * 200}, ${colorIntensity * 200}, ${255})`; // 発火に応じて青みがかった色に
            nodeElement.setAttribute('r', nodeRadius * (1 + activation * 0.5)); // 活性化に応じて半径を少し大きく
        }
    }

    // 隠れ層2 (output_layer2_relu)
    const layer2Values = intermediateOutputs.output_layer2_relu.data;
    for (let i = 0; i < layer2Values.length; i++) {
        const nodeElement = document.getElementById(`node-L2-N${i}`); // Layer 2 (Hidden2)
        if (nodeElement) {
            const activation = layer2Values[i];
            const colorIntensity = Math.min(1, Math.max(0, activation));
            nodeElement.style.fill = `rgb(${255}, ${colorIntensity * 200}, ${colorIntensity * 200})`; // 発火に応じて赤みがかった色に
            nodeElement.setAttribute('r', nodeRadius * (1 + activation * 0.5));
        }
    }

    // 隠れ層3 (output_layer3_relu)
    const layer3Values = intermediateOutputs.output_layer3_relu.data;
    for (let i = 0; i < layer3Values.length; i++) {
        const nodeElement = document.getElementById(`node-L3-N${i}`); // Layer 3 (Hidden3)
        if (nodeElement) {
            const activation = layer3Values[i];
            const colorIntensity = Math.min(1, Math.max(0, activation));
            nodeElement.style.fill = `rgb(${colorIntensity * 200}, ${255}, ${colorIntensity * 200})`; // 発火に応じて緑みがかった色に
            nodeElement.setAttribute('r', nodeRadius * (1 + activation * 0.5));
        }
    }

    // 出力層 (output_final)
    const outputValues = finalOutputProbabilities;
    for (let i = 0; i < outputValues.length; i++) {
        const nodeElement = document.getElementById(`node-L4-N${i}`); // Layer 4 (Output)
        if (nodeElement) {
            const probability = outputValues[i];
            const colorIntensity = probability; // 確率をそのまま強度として使用
            nodeElement.style.fill = `rgb(${colorIntensity * 255}, 0, 0)`; // 確率が高いほど赤く
            nodeElement.setAttribute('r', nodeRadius * (1 + probability * 0.8)); // 確率が高いほど大きく
        }
    }
}

// ページロード時にNN構造を描画
window.addEventListener('load', drawNN);

// 推論処理の修正: 中間層の出力を取得し、可視化関数に渡す
predictButton.addEventListener('click', async () => {
    if (!session) {
        predictionResultSpan.textContent = 'モデルがまだ読み込まれていません。';
        return;
    }

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const features = extractFeatures(imageData);
    const inputTensor = new onnx.Tensor(features, 'float32', [1, 4]);

    try {
        // 推論を実行し、すべての出力ノードの値を取得
        // ONNXモデルのエクスポート時に指定した output_names を参照
        const outputMap = await session.run([inputTensor], ['output_final', 'output_layer1_relu', 'output_layer2_relu', 'output_layer3_relu']);

        const intermediateOutputs = {
            output_layer1_relu: outputMap.get('output_layer1_relu'),
            output_layer2_relu: outputMap.get('output_layer2_relu'),
            output_layer3_relu: outputMap.get('output_layer3_relu')
        };

        const finalOutputTensor = outputMap.get('output_final');
        const logits = Array.from(finalOutputTensor.data);
        const expLogits = logits.map(Math.exp);
        const sumExpLogits = expLogits.reduce((sum, val) => sum + val, 0);
        const probabilities = expLogits.map(val => val / sumExpLogits);

        let maxProb = -1;
        let predictedClass = -1;
        for (let i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                predictedClass = i;
            }
        }

        predictionResultSpan.textContent = predictedClass;
        probSpans[0].textContent = `${(probabilities[0] * 100).toFixed(2)}%`;
        probSpans[1].textContent = `${(probabilities[1] * 100).toFixed(2)}%`;
        probSpans[2].textContent = `${(probabilities[2] * 100).toFixed(2)}%`;

        // NN構造図の可視化を更新
        updateNNVisualization(features, intermediateOutputs, probabilities);  // outputMapには中間層のテンソルが含まれる

    } catch (error) {
        console.error('推論中にエラーが発生しました:', error);
        predictionResultSpan.textContent = '推論エラー';
    }
});

// ====== 特徴量抽出関数 (JavaScript版) ======
// Pythonのextract_features関数をJavaScriptで再現
function extractFeatures(imageData) {
    const pixels = new Float32Array(28 * 28);
    // ImageDataはRGBAなので、Rチャネルだけをグレースケールとして使用 (28x28に縮小後)
    for (let i = 0; i < 28 * 28; i++) {
        // 画像データは0-255だが、Python側は0-1で正規化されているのでここで255で割る
        pixels[i] = imageData.data[i * 4] / 255.0; 
    }

    // 1. 総ピクセル密度
    let totalPixelValue = pixels.reduce((sum, val) => sum + val, 0);
    let totalPixelDensity = totalPixelValue / (28 * 28);

    // 2. 上半分と下半分のピクセル密度の差
    let upperHalfSum = 0;
    let lowerHalfSum = 0;
    for (let r = 0; r < 28; r++) {
        for (let c = 0; c < 28; c++) {
            const pixel = pixels[r * 28 + c];
            if (r < 14) {
                upperHalfSum += pixel;
            } else {
                lowerHalfSum += pixel;
            }
        }
    }
    let upperHalfDensity = upperHalfSum / (14 * 28);
    let lowerHalfDensity = lowerHalfSum / (14 * 28);
    let topBottomDensityDiff = upperHalfDensity - lowerHalfDensity;

    // 3. 垂直方向の重心
    let verticalCentroid;
    if (totalPixelValue < 1e-6) {
        verticalCentroid = 0.5;
    } else {
        let weightedSumY = 0;
        for (let r = 0; r < 28; r++) {
            for (let c = 0; c < 28; c++) {
                weightedSumY += r * pixels[r * 28 + c];
            }
        }
        verticalCentroid = weightedSumY / totalPixelValue;
        verticalCentroid = verticalCentroid / 27.0; // 0-27を0-1に正規化
    }

    // 4. 水平方向の交差数
    let crossingCount = 0;
    const threshold = 0.05;
    for (let r = 0; r < 28; r++) {
        let isDrawing = pixels[r * 28] > threshold;
        for (let c = 1; c < 28; c++) {
            const nextIsDrawing = pixels[r * 28 + c] > threshold;
            if (isDrawing !== nextIsDrawing) {
                crossingCount++;
            }
            isDrawing = nextIsDrawing;
        }
    }
    const maxPossibleCrossings = 112;
    let horizontalCrossingCount = crossingCount / maxPossibleCrossings;

    // 特徴量を配列にまとめる
    let features = [
        totalPixelDensity,
        topBottomDensityDiff,
        verticalCentroid,
        horizontalCrossingCount
    ];

    return new Float32Array(features);
}
