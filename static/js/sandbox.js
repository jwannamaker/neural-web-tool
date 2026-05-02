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
    lossHistory = [];

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

    const decoder = new TextDecoder();
    let buffer = '';

    for await (const chunk of res.body) {
        buffer += decoder.decode(chunk, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop();

        for (const event of events) {
            const line = event.trim();
            if (!line.startsWith('data: ')) continue;
            const data = JSON.parse(line.slice(6));
            if (data.done) {
                stopProgress();
                document.getElementById('status').textContent = data.message;
                drawNetwork(getLayerSizesFromUI());
            } else {
                lossHistory.push(data.loss);
                drawLossGraph(lossHistory);
            }
        }
    }
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

// ─── Loss graph canvas ────────────────────────────────────────────────────────

const lossCanvas = document.getElementById('loss-canvas');
const lossCtx    = lossCanvas.getContext('2d');
let   lossHistory = [];

function drawLossGraph(losses) {
    const W = lossCanvas.offsetWidth;
    const H = lossCanvas.offsetHeight;

    lossCanvas.width  = W * devicePixelRatio;
    lossCanvas.height = H * devicePixelRatio;
    lossCtx.scale(devicePixelRatio, devicePixelRatio);

    lossCtx.fillStyle = '#ffffff';
    lossCtx.fillRect(0, 0, W, H);

    if (losses.length === 0) return;

    const pad    = { top: 20, right: 20, bottom: 44, left: 64 };
    const graphW = W - pad.left - pad.right;
    const graphH = H - pad.top  - pad.bottom;

    const minLoss   = Math.min(...losses);
    const maxLoss   = Math.max(...losses);
    const lossRange = maxLoss - minLoss || 1;

    const toX = i => pad.left + (losses.length === 1 ? graphW / 2 : (i / (losses.length - 1)) * graphW);
    const toY = v => pad.top  + (1 - (v - minLoss) / lossRange) * graphH;

    // Horizontal grid lines + Y-axis labels
    const gridLines = 4;
    for (let g = 0; g <= gridLines; g++) {
        const y     = pad.top + (g / gridLines) * graphH;
        const value = maxLoss - (g / gridLines) * lossRange;

        lossCtx.strokeStyle = '#e0e0e0';
        lossCtx.lineWidth   = 1;
        lossCtx.beginPath();
        lossCtx.moveTo(pad.left, y);
        lossCtx.lineTo(pad.left + graphW, y);
        lossCtx.stroke();

        lossCtx.fillStyle    = ELLIPSIS_COLOR;
        lossCtx.font         = '11px sans-serif';
        lossCtx.textAlign    = 'right';
        lossCtx.textBaseline = 'middle';
        lossCtx.fillText(value.toFixed(3), pad.left - 4, y);
    }

    // X-axis epoch labels
    const labelStep = Math.ceil(losses.length / 10);
    lossCtx.fillStyle    = ELLIPSIS_COLOR;
    lossCtx.font         = '11px sans-serif';
    lossCtx.textAlign    = 'center';
    lossCtx.textBaseline = 'top';
    losses.forEach((_, i) => {
        if (i % labelStep === 0 || i === losses.length - 1)
            lossCtx.fillText(i + 1, toX(i), pad.top + graphH + 4);
    });

    // Axes
    lossCtx.strokeStyle = '#808080';
    lossCtx.lineWidth   = 1;
    lossCtx.beginPath();
    lossCtx.moveTo(pad.left, pad.top);
    lossCtx.lineTo(pad.left, pad.top + graphH);
    lossCtx.lineTo(pad.left + graphW, pad.top + graphH);
    lossCtx.stroke();

    // Axis titles
    lossCtx.fillStyle    = ELLIPSIS_COLOR;
    lossCtx.font         = '12px sans-serif';

    lossCtx.textAlign    = 'center';
    lossCtx.textBaseline = 'bottom';
    lossCtx.fillText('Epoch', pad.left + graphW / 2, H - 2);

    lossCtx.save();
    lossCtx.translate(12, pad.top + graphH / 2);
    lossCtx.rotate(-Math.PI / 2);
    lossCtx.textAlign    = 'center';
    lossCtx.textBaseline = 'top';
    lossCtx.fillText('Loss', 0, 0);
    lossCtx.restore();

    // Loss line
    lossCtx.strokeStyle = NEURON_STROKE;
    lossCtx.lineWidth   = 2;
    lossCtx.beginPath();
    losses.forEach((loss, i) => {
        if (i === 0) lossCtx.moveTo(toX(i), toY(loss));
        else         lossCtx.lineTo(toX(i), toY(loss));
    });
    lossCtx.stroke();

    // Data points
    lossCtx.fillStyle = NEURON_STROKE;
    losses.forEach((loss, i) => {
        lossCtx.beginPath();
        lossCtx.arc(toX(i), toY(loss), 3, 0, Math.PI * 2);
        lossCtx.fill();
    });
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

window.addEventListener('resize', () => {
    drawNetwork(getLayerSizesFromUI());
    drawLossGraph(lossHistory);
});

// ─── Initialize ──────────────────────────────────────────────────────────

addLayerConfigRow(784, 'relu', true, false);
addLayerConfigRow(128, 'relu', false, false);
addLayerConfigRow(64, 'relu', false, false);
addLayerConfigRow(10, 'linear', false, true);
drawNetwork([784, 128, 64, 10]);
drawLossGraph([]);

