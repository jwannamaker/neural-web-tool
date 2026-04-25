document.addEventListener("DOMContentLoaded", () => {
    const trainForm = document.getElementById("train-form");
    const trainBtn = document.getElementById("train-btn");
    const evalBtn = document.getElementById("eval-btn");

    const statusArea = document.getElementById("status-area");
    const statusMsg = document.getElementById("status-msg");
    const accuracyResult = document.getElementById("accuracy-result");
    const classMetricsTable = document.getElementById("class-metrics");
    const metricsBody = document.getElementById("metrics-body");

    const canvas = document.getElementById("drawing-canvas");
    const ctx = canvas.getContext("2d");
    const predictBtn = document.getElementById("predict-btn");
    const clearBtn = document.getElementById("clear-btn");
    const predictionResults = document.getElementById("prediction-results");
    const predictionStatus = document.getElementById("prediction-status");
    const errorDialog = document.getElementById("error-dialog");
    const historyBody = document.getElementById("history-body");
    const clearHistoryBtn = document.getElementById("clear-history-btn");

    let predictionHistory = [];

    // Animation & Progress Bar Elements
    const animArea = document.getElementById("animArea");
    const progressBar = document.getElementById("progressBar");
    const folderTemplate = document.getElementById("fileIcon").innerHTML;

    // Create Progress Blocks
    const numBlocks = 24;
    for (let i = 0; i < numBlocks; i++) {
        const block = document.createElement("div");
        block.className = "progress-block";
        progressBar.appendChild(block);
    }
    const blocks = progressBar.querySelectorAll(".progress-block");

    function updateProgressBlocks(percent) {
        const blocksToShow = Math.floor((percent / 100) * numBlocks);
        blocks.forEach((block, index) => {
            block.style.display = index < blocksToShow ? "block" : "none";
        });
    }

    let animationInterval = null;

    function spawnFolder() {
        const folder = document.createElement("div");
        folder.className = "flying-folder";
        folder.innerHTML = folderTemplate;

        const yOffset = Math.random() * 20 - 10;
        folder.style.top = `${25 + yOffset}px`;
        folder.style.left = '40px';

        animArea.appendChild(folder);

        // Force reflow
        folder.getBoundingClientRect();

        const animDuration = 2000 + Math.random() * 500;
        folder.style.transition = `all ${animDuration}ms linear`;
        folder.style.left = `${animArea.offsetWidth - 60}px`;
        folder.style.top = `${25 + (Math.random() * 10 - 5)}px`;

        setTimeout(() => {
            if (folder.parentNode) {
                folder.parentNode.removeChild(folder);
            }
        }, animDuration);
    }

    function startTrainingAnimation() {
        if (animationInterval) clearInterval(animationInterval);
        animationInterval = setInterval(spawnFolder, 4000);
        updateProgressBlocks(0);
        spawnFolder(); // Spawn one immediately
    }

    function stopTrainingAnimation() {
        if (animationInterval) {
            clearInterval(animationInterval);
            animationInterval = null;
        }
        updateProgressBlocks(100);
    }

    function setStatus(msg, loading = false) {
        statusMsg.textContent = msg;
        statusMsg.style.color = "#334155";
        if (loading) {
            accuracyResult.style.display = "none";
        }
    }

    function setError(msg) {
        statusMsg.textContent = "Error: " + msg;
        statusMsg.style.color = "#ef4444";
        accuracyResult.style.display = "none";
        classMetricsTable.style.display = "none";
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
        setStatus("Training in progress...", true);
        startTrainingAnimation();

        // Simulate progress for the block bar since training is sync on backend
        let fakeProgress = 0;
        const progressInterval = setInterval(() => {
            // Asymptotic progress: distance to 99% gets smaller over time
            const remaining = 99 - fakeProgress;
            fakeProgress += (remaining * 0.05) + (Math.random() * 0.5);
            if (fakeProgress > 99) fakeProgress = 99;
            updateProgressBlocks(fakeProgress);
        }, 500);

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
            }
        } catch (err) {
            setError(err.message);
        } finally {
            trainBtn.disabled = false;
            stopTrainingAnimation();
            clearInterval(progressInterval);
            updateProgressBlocks(100);
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

                // Render per-class metrics table
                metricsBody.innerHTML = "";
                classMetricsTable.style.display = "table";
                Object.entries(data.class_accuracies).forEach(([digit, acc]) => {
                    const row = document.createElement("tr");
                    row.innerHTML = `
                        <td><strong>Digit ${digit}</strong></td>
                        <td>${acc.toFixed(1)}%</td>
                    `;
                    metricsBody.appendChild(row);
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
                if (data.error && data.error.includes("No network")) {
                    errorDialog.style.display = "block";
                    predictionStatus.textContent = "Error: System not initialized.";
                } else {
                    predictionStatus.textContent = "Error: " + (data.error || "Prediction failed.");
                }
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

                // Add to History
                const topPrediction = data.predictions[0];
                addToHistory(imageData, topPrediction.digit, (topPrediction.confidence * 100).toFixed(1));
            }
        } catch (err) {
            predictionStatus.textContent = "Error: " + err.message;
        }
    });

    function addToHistory(imageSrc, digit, confidence) {
        const id = Date.now();
        const entry = { id, imageSrc, digit, confidence, feedback: null };
        predictionHistory.unshift(entry); // Add to start
        renderHistory();
    }

    function renderHistory() {
        historyBody.innerHTML = "";
        predictionHistory.forEach(entry => {
            const row = document.createElement("tr");

            let feedbackHtml = "";
            if (entry.feedback === "correct") {
                feedbackHtml = '<span class="feedback-correct">✓ Right</span>';
            } else if (entry.feedback === "wrong") {
                feedbackHtml = '<span class="feedback-wrong">✗ Wrong</span>';
            } else {
                feedbackHtml = `
                    <button class="btn-win feedback-btn" onclick="provideFeedback(${entry.id}, 'correct')">Right</button>
                    <button class="btn-win feedback-btn" onclick="provideFeedback(${entry.id}, 'wrong')">Wrong</button>
                `;
            }

            row.innerHTML = `
                <td><img src="${entry.imageSrc}" class="history-thumbnail"></td>
                <td><strong>Digit ${entry.digit}</strong></td>
                <td>${entry.confidence}%</td>
                <td>${feedbackHtml}</td>
            `;
            historyBody.appendChild(row);
        });
    }

    window.provideFeedback = (id, type) => {
        const entry = predictionHistory.find(e => e.id === id);
        if (entry) {
            entry.feedback = type;
            renderHistory();
        }
    };

    clearHistoryBtn.addEventListener("click", () => {
        predictionHistory = [];
        renderHistory();
    });
});
