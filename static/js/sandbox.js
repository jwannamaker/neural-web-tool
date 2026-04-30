// ─── Layer config UI ─────────────────────────────────────────────────────────

function addLayerConfigRow(size = 64, activation = 'relu', isInput = false, isOutput = false, insertBefore = null) {
    const container = document.getElementById('layer-config');
    const row = document.createElement('div');
    row.className = 'layer-config-row';

    row.innerHTML = `
        <input type="number" class="layer-size" value="${size}" min="1" ${isInput ? 'readonly' : ''}>
        ${isOutput
            ? `<span class="layer-activation-label">Linear (output)</span>`
            : `<select class="layer-activation">
                   <option value="relu"    ${activation === 'relu' ? 'selected' : ''}>ReLU</option>
                   <option value="sigmoid" ${activation === 'sigmoid' ? 'selected' : ''}>Sigmoid</option>
                   <option value="tanh"    ${activation === 'tanh' ? 'selected' : ''}>Tanh</option>
                   <option value="linear"  ${activation === 'linear' ? 'selected' : ''}>Linear</option>
               </select>
               ${!isInput ? '<button class="remove-row-button">Remove</button>' : ''}`
        }
    `;

    if (!isInput && !isOutput) {
        row.querySelector('.remove-row-button').addEventListener('click', () => row.remove());
    }
    if (insertBefore) {
        container.insertBefore(row, insertBefore);
    } else {
        container.appendChild(row);
    }
}

function getLayerSizesFromUI() {
    return [...document.querySelectorAll('.layer-size')]
        .map(input => parseInt(input.value));
}

function getActivationsFromUI() {
    return [...document.querySelectorAll('.layer-activation')]
        .map(select => select.value);
}

// ─── API calls ────────────────────────────────────────────────────────────────

let progressInterval = null;

function startProgress() {
    const bar = document.getElementById('progress-bar');
    const blocks = bar.querySelectorAll('.progress-block');
    bar.classList.add('running');
    let i = 0;
    blocks.forEach(b => { b.style.opacity = 0; });
    progressInterval = setInterval(() => {
        const n = blocks.length;
        blocks.forEach((b, idx) => {
            b.style.opacity = ((idx - i % n + n) % n) < 3 ? 1 : 0;
        });
        i++;
    }, 80);
}

function stopProgress() {
    const bar = document.getElementById('progress-bar');
    const blocks = bar.querySelectorAll('.progress-block');
    clearInterval(progressInterval);
    progressInterval = null;
    blocks.forEach(b => { b.style.opacity = 0; });
    bar.classList.remove('running');
}

async function trainNetwork() {
    document.getElementById('status').textContent = 'Training...';
    startProgress();

    const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            layer_sizes: getLayerSizesFromUI(),
            activations: getActivationsFromUI(),
            loss: document.getElementById('loss').value,
            optimizer: document.getElementById('optimizer').value,
            lr: parseFloat(document.getElementById('lr').value),
            epochs: parseInt(document.getElementById('epochs').value),
            batch_size: parseInt(document.getElementById('batch').value)
        })
    });

    stopProgress();
    const data = await res.json();
    document.getElementById('status').textContent = data.message || data.error;
    drawNetwork(getLayerSizesFromUI());
}

async function evaluateNetwork() {
    document.getElementById('status').textContent = 'Evaluating...';

    const res = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batch_size: parseInt(document.getElementById('batch').value) })
    });

    const data = await res.json();
    document.getElementById('status').textContent = 'Done';
    document.getElementById('accuracy-display').textContent =
        data.accuracy !== undefined
            ? `Accuracy: ${data.accuracy.toFixed(1)}%`
            : data.error;
}

async function predictDrawing() {
    const imageDataURL = drawCanvas.toDataURL('image/png');

    const res = await fetch('/api/predict_image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageDataURL })
    });

    const data = await res.json();
    if (data.error) {
        document.getElementById('prediction-result').textContent = data.error;
        return;
    }

    const top = data.predictions[0];
    document.getElementById('prediction-result').textContent =
        `Digit: ${top.digit} — ${(top.confidence * 100).toFixed(1)}% confidence`;
}

// ─── Network canvas ───────────────────────────────────────────────────────────

const netCanvas = document.getElementById('network-canvas');
const netCtx = netCanvas.getContext('2d');

const MAX_VISIBLE = 10;
const SHOW_TOP = 5;
const NEURON_R = 10;

function getColors() {
    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    return {
        neuronFill: isDark ? '#1D3A5F' : '#E6F1FB',
        neuronStroke: isDark ? '#378ADD' : '#185FA5',
        ellipsis: isDark ? '#888780' : '#5F5E5A',
        conn: isDark ? '#378ADD0F' : '#185FA50F',
    };
}

function visibleNeurons(count) {
    if (count <= MAX_VISIBLE) {
        return { indices: Array.from({ length: count }, (_, i) => i), hasEllipsis: false };
    }
    const top = Array.from({ length: SHOW_TOP }, (_, i) => i);
    const bot = Array.from({ length: SHOW_TOP }, (_, i) => count - SHOW_TOP + i);
    return { indices: [...top, ...bot], hasEllipsis: true };
}

function realSlots(vis) {
    if (!vis.hasEllipsis) return vis.indices.map((_, i) => i);
    return [
        ...Array.from({ length: SHOW_TOP }, (_, i) => i),
        ...Array.from({ length: SHOW_TOP }, (_, i) => SHOW_TOP + 1 + i)
    ];
}

function getNeuronY(vis, slot, H) {
    const total = vis.hasEllipsis ? MAX_VISIBLE + 1 : vis.indices.length;
    const spacing = Math.min(30, (H - 60) / Math.max(total, 1));
    const startY = H / 2 - (total - 1) * spacing / 2;
    return startY + slot * spacing;
}

function drawNetwork(layers) {
    const W = netCanvas.offsetWidth;
    netCanvas.width = W * devicePixelRatio;
    netCanvas.height = netCanvas.offsetHeight * devicePixelRatio;
    netCtx.scale(devicePixelRatio, devicePixelRatio);
    const H = netCanvas.offsetHeight;
    const C = getColors();

    netCtx.clearRect(0, 0, W, H);

    const layerXs = layers.map((_, i) => 60 + (i / (layers.length - 1)) * (W - 120));
    const layerVis = layers.map(n => visibleNeurons(n));

    // Connections first (drawn underneath neurons)
    netCtx.strokeStyle = C.conn;
    netCtx.lineWidth = 0.5;
    for (let li = 0; li < layers.length - 1; li++) {
        const slotsA = realSlots(layerVis[li]);
        const slotsB = realSlots(layerVis[li + 1]);
        for (const sa of slotsA) {
            for (const sb of slotsB) {
                netCtx.beginPath();
                netCtx.moveTo(layerXs[li], getNeuronY(layerVis[li], sa, H));
                netCtx.lineTo(layerXs[li + 1], getNeuronY(layerVis[li + 1], sb, H));
                netCtx.stroke();
            }
        }
    }

    // Neurons on top
    for (let li = 0; li < layers.length; li++) {
        const vis = layerVis[li];
        const x = layerXs[li];
        const slots = realSlots(vis);

        if (vis.hasEllipsis) {
            netCtx.fillStyle = C.ellipsis;
            netCtx.font = '500 14px sans-serif';
            netCtx.textAlign = 'center';
            netCtx.textBaseline = 'middle';
            netCtx.fillText('···', x, getNeuronY(vis, SHOW_TOP, H));
        }

        for (const slot of slots) {
            const y = getNeuronY(vis, slot, H);
            netCtx.beginPath();
            netCtx.arc(x, y, NEURON_R, 0, Math.PI * 2);
            netCtx.fillStyle = C.neuronFill;
            netCtx.fill();
            netCtx.strokeStyle = C.neuronStroke;
            netCtx.lineWidth = 1;
            netCtx.stroke();
        }
    }
}

// ─── Draw canvas (digit input) ────────────────────────────────────────────────

const drawCanvas = document.getElementById('draw-canvas');
const drawCtx = drawCanvas.getContext('2d');
let isDrawing = false;

function clearDrawCanvas() {
    drawCtx.fillStyle = '#000000';
    drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
}

drawCtx.lineWidth = 18;
drawCtx.lineCap = 'round';
drawCtx.strokeStyle = '#ffffff';
clearDrawCanvas();

drawCanvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    drawCtx.beginPath();
    drawCtx.moveTo(e.offsetX, e.offsetY);
});
drawCanvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    drawCtx.lineTo(e.offsetX, e.offsetY);
    drawCtx.stroke();
});
drawCanvas.addEventListener('mouseup', () => { isDrawing = false; });
drawCanvas.addEventListener('mouseleave', () => { isDrawing = false; });

// ─── Slider display updates ───────────────────────────────────────────────────

document.getElementById('lr').addEventListener('input', (e) => {
    document.getElementById('lr-display').textContent = parseFloat(e.target.value).toFixed(4);
});
document.getElementById('epochs').addEventListener('input', (e) => {
    document.getElementById('epochs-display').textContent = e.target.value;
});
document.getElementById('batch').addEventListener('input', (e) => {
    document.getElementById('batch-display').textContent = e.target.value;
});

// ─── Event listeners ──────────────────────────────────────────────────────────

// document.addEventListener('DOMContentLoaded', () => {

document.getElementById('add-layer-button').addEventListener('click', () => {
    const rows = document.querySelectorAll('.layer-config-row');
    const outputRow = rows[rows.length - 1];
    addLayerConfigRow(64, 'relu', false, outputRow);
});

document.getElementById('train-button').addEventListener('click', trainNetwork);
document.getElementById('evaluate-button').addEventListener('click', evaluateNetwork);
document.getElementById('predict-button').addEventListener('click', predictDrawing);
document.getElementById('clear-button').addEventListener('click', clearDrawCanvas);

window.addEventListener('resize', () => drawNetwork(getLayerSizesFromUI()));

// ─── Initialize ──────────────────────────────────────────────────────────

addLayerConfigRow(784, 'relu', true, false);
addLayerConfigRow(128, 'relu', false, false);
addLayerConfigRow(64, 'relu', false, false);
addLayerConfigRow(10, 'linear', false, true);
drawNetwork([784, 128, 64, 10]);

