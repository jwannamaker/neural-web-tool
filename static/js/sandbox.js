// ─── Layer config UI ─────────────────────────────────────────────────────────

function addLayerConfigRow(size = 10, activation = 'relu', isInput = false, isOutput = false, insertBefore = null) {
    const container = document.getElementById('layer-config');
    const row = document.createElement('div');
    row.className = 'layer-config-row';

    row.innerHTML = `
        <label class="row-label">''</label>
        <input type="number" class="layer-size" value="${size}" min="1" ${isInput ? 'readonly' : ''}>
        ${isOutput
            ? `<label class="layer-activation-label">Linear</label>`
            : `<select class="layer-activation">
                   <option value="relu"    ${activation === 'relu' ? 'selected' : ''}>ReLU</option>
                   <option value="sigmoid" ${activation === 'sigmoid' ? 'selected' : ''}>Sigmoid</option>
                   <option value="tanh"    ${activation === 'tanh' ? 'selected' : ''}>Tanh</option>
                   <option value="linear"  ${activation === 'linear' ? 'selected' : ''}>Linear</option>
               </select>
               ${!isInput ? '<button class="remove-row-button">X</button>' : ''}`}
    `;
    if (!isInput && !isOutput) {
        row.querySelector('.remove-row-button').addEventListener('click', () => {
            row.remove();
            relabelRows(); // renumber after removal
        });
    }

    if (insertBefore) {
        container.insertBefore(row, insertBefore);
    } else {
        container.appendChild(row);
    }

    relabelRows(); // renumber after addition
}

function relabelRows() {
    const rows = document.querySelectorAll('.layer-config-row');
    let hiddenCount = 0;
    rows.forEach((row, i) => {
        const label = row.querySelector('.row-label');
        if (i === 0) {
            label.textContent = 'Input';
        } else if (i === rows.length - 1) {
            label.textContent = 'Output';
        } else {
            hiddenCount++;
            label.textContent = `Hidden ${hiddenCount}`;
        }
    });
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

const NEURON_FILL = '#E6F1FB';
const NEURON_STROKE = '#185FA5';
const CONN_COLOR = '#185FA520';
const ELLIPSIS_COLOR = '#5F5E5A';

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

function getNeuronY(vis, slot, canvasHeight) {
    const totalSlots = vis.hasEllipsis ? MAX_VISIBLE + 1 : vis.indices.length;
    const vertPadding = 60;
    const maxSpacing = 30;
    const spacing = Math.min(maxSpacing, (canvasHeight - vertPadding) / Math.max(totalSlots, 1));
    const columnTop = (canvasHeight - (totalSlots - 1) * spacing) / 2;
    return columnTop + slot * spacing;
}

function drawNetwork(layers) {
    const W = netCanvas.offsetWidth;
    const H = netCanvas.offsetHeight;

    netCanvas.width = W * devicePixelRatio;
    netCanvas.height = H * devicePixelRatio;
    netCtx.scale(devicePixelRatio, devicePixelRatio);

    // White Win95-style background
    netCtx.fillStyle = '#ffffff';
    netCtx.fillRect(0, 0, W, H);

    if (layers.length < 2) return;

    const hPad = 60;
    const layerXs = layers.map((_, i) => hPad + (i / (layers.length - 1)) * (W - 2 * hPad));
    const layerVis = layers.map(count => visibleNeurons(count));

    // Connections drawn first, behind neurons
    netCtx.strokeStyle = CONN_COLOR;
    netCtx.lineWidth = 0.5;
    for (let li = 0; li < layers.length - 1; li++) {
        const fromSlots = realSlots(layerVis[li]);
        const toSlots = realSlots(layerVis[li + 1]);
        for (const fromSlot of fromSlots) {
            const x1 = layerXs[li];
            const y1 = getNeuronY(layerVis[li], fromSlot, H);
            for (const toSlot of toSlots) {
                netCtx.beginPath();
                netCtx.moveTo(x1, y1);
                netCtx.lineTo(layerXs[li + 1], getNeuronY(layerVis[li + 1], toSlot, H));
                netCtx.stroke();
            }
        }
    }

    // Neurons and ellipses drawn on top of connections
    for (let li = 0; li < layers.length; li++) {
        const vis = layerVis[li];
        const x = layerXs[li];
        const slots = realSlots(vis);

        if (vis.hasEllipsis) {
            netCtx.fillStyle = ELLIPSIS_COLOR;
            netCtx.font = '500 14px sans-serif';
            netCtx.textAlign = 'center';
            netCtx.textBaseline = 'middle';
            netCtx.fillText('···', x, getNeuronY(vis, SHOW_TOP, H));
        }

        for (const slot of slots) {
            const y = getNeuronY(vis, slot, H);
            netCtx.beginPath();
            netCtx.arc(x, y, NEURON_R, 0, Math.PI * 2);
            netCtx.fillStyle = NEURON_FILL;
            netCtx.fill();
            netCtx.strokeStyle = NEURON_STROKE;
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
    addLayerConfigRow(64, 'relu', false, false, outputRow);
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

