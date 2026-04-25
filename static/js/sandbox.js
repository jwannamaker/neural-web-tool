document.addEventListener("DOMContentLoaded", () => {
    const trainForm = document.getElementById("train-form");
    const trainBtn = document.getElementById("train-btn");
    const evalBtn = document.getElementById("eval-btn");

    const statusArea = document.getElementById("status-area");
    const statusMsg = document.getElementById("status-msg");
    const spinner = document.getElementById("loading-spinner");
    const accuracyResult = document.getElementById("accuracy-result");
    const classMetrics = document.getElementById("class-metrics");

    const canvas = document.getElementById("drawing-canvas");
    const ctx = canvas.getContext("2d");
    const predictBtn = document.getElementById("predict-btn");
    const clearBtn = document.getElementById("clear-btn");
    const predictionResults = document.getElementById("prediction-results");
    const predictionStatus = document.getElementById("prediction-status");

    function setStatus(msg, loading = false) {
        statusMsg.textContent = msg;
        statusMsg.style.color = "#334155";
        if (loading) {
            spinner.style.display = "block";
            accuracyResult.style.display = "none";
        } else {
            spinner.style.display = "none";
        }
    }

    function setError(msg) {
        statusMsg.textContent = "Error: " + msg;
        statusMsg.style.color = "#ef4444";
        spinner.style.display = "none";
        accuracyResult.style.display = "none";
        classMetrics.style.display = "none";
    }

    // Canvas Logic
    let isDrawing = false;
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.strokeStyle = "white";
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }

    function startDrawing(e) {
        isDrawing = true;
        const pos = getPos(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        e.preventDefault();
    }

    function draw(e) {
        if (!isDrawing) return;
        const pos = getPos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        e.preventDefault();
    }

    function stopDrawing() {
        isDrawing = false;
    }

    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseleave", stopDrawing);

    canvas.addEventListener("touchstart", startDrawing);
    canvas.addEventListener("touchmove", draw);
    canvas.addEventListener("touchend", stopDrawing);

    clearBtn.addEventListener("click", () => {
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        predictionResults.style.display = "none";
        predictionStatus.style.display = "block";
        predictionStatus.textContent = "Draw something and click predict to see results.";
    });

    trainForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Gather data
        const layersRaw = document.getElementById("layers").value;
        const activationsRaw = document.getElementById("activations").value;

        const layerSizes = layersRaw.split(",").map(s => parseInt(s.trim())).filter(n => !isNaN(n));
        const activations = activationsRaw.split(",").map(s => s.trim()).filter(s => s);

        const payload = {
            layer_sizes: layerSizes,
            activations: activations,
            loss: document.getElementById("loss").value,
            optimizer: document.getElementById("optimizer").value,
            lr: parseFloat(document.getElementById("lr").value),
            epochs: parseInt(document.getElementById("epochs").value),
            batch_size: parseInt(document.getElementById("batch_size").value)
        };

        // UI Updates
        trainBtn.disabled = true;
        evalBtn.disabled = true;
        setStatus("Training in progress... This may take a few moments depending on the number of epochs.", true);

        try {
            const response = await fetch("/api/train", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                setError(data.error || "Training failed.");
            } else {
                setStatus(data.message);
                evalBtn.disabled = false; // Enable evaluation
                predictBtn.disabled = false; // Enable prediction
            }
        } catch (err) {
            setError(err.message);
        } finally {
            trainBtn.disabled = false;
        }
    });

    evalBtn.addEventListener("click", async () => {
        evalBtn.disabled = true;
        trainBtn.disabled = true;
        setStatus("Evaluating model on test dataset...", true);

        const batchSize = parseInt(document.getElementById("batch_size").value);

        try {
            const response = await fetch("/api/evaluate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ batch_size: batchSize })
            });

            const data = await response.json();

            if (!response.ok) {
                setError(data.error || "Evaluation failed.");
            } else {
                setStatus("Evaluation Complete!");
                accuracyResult.textContent = data.accuracy.toFixed(2) + "%";
                accuracyResult.style.display = "block";

                // Render per-class metrics
                classMetrics.innerHTML = "";
                classMetrics.style.display = "grid";
                Object.entries(data.class_accuracies).forEach(([digit, acc]) => {
                    const item = document.createElement("div");
                    item.className = "class-metric-item";
                    item.innerHTML = `<span class="class-metric-digit">${digit}</span> ${acc.toFixed(0)}%`;
                    classMetrics.appendChild(item);
                });
            }
        } catch (err) {
            setError(err.message);
        } finally {
            evalBtn.disabled = false;
            trainBtn.disabled = false;
        }
    });

    predictBtn.addEventListener("click", async () => {
        predictionStatus.textContent = "Analyzing...";
        predictionResults.style.display = "none";
        
        const imageData = canvas.toDataURL("image/png");

        try {
            const response = await fetch("/api/predict_image", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: imageData })
            });

            const data = await response.json();

            if (!response.ok) {
                predictionStatus.textContent = "Error: " + (data.error || "Prediction failed.");
            } else {
                predictionStatus.style.display = "none";
                predictionResults.innerHTML = "";
                predictionResults.style.display = "block";

                data.predictions.forEach(pred => {
                    const item = document.createElement("div");
                    item.className = "prediction-item";
                    const confPercent = (pred.confidence * 100).toFixed(1);
                    item.innerHTML = `
                        <div class="prediction-digit">${pred.digit}</div>
                        <div class="prediction-bar-container">
                            <div class="prediction-bar" style="width: ${confPercent}%"></div>
                        </div>
                        <div class="prediction-conf">${confPercent}%</div>
                    `;
                    predictionResults.appendChild(item);
                });
            }
        } catch (err) {
            predictionStatus.textContent = "Error: " + err.message;
        }
    });
});
