document.addEventListener("DOMContentLoaded", () => {
    const terminalContent = document.getElementById("terminal-content");
    const terminalWindow = document.getElementById("terminal-window");
    const refreshBtn = document.getElementById("refresh-logs-btn");
    const autoScrollBtn = document.getElementById("auto-scroll-btn");

    let autoScroll = true;
    let lastLogCount = 0;

    function getLevelClass(level) {
        switch(level) {
            case "SUCCESS": return "terminal-success";
            case "DEBUG": return "terminal-debug";
            case "ERROR": return "terminal-error";
            default: return "terminal-info";
        }
    }

    async function fetchLogs() {
        try {
            const response = await fetch("/api/logs_data");
            const data = await response.json();
            
            if (data.logs.length > lastLogCount || data.logs.length === 0) {
                // Completely rewrite logs for simplicity, or just append new ones.
                // Since array clears on new training, rewriting is safer.
                terminalContent.innerHTML = "";
                data.logs.forEach(log => {
                    const line = document.createElement("div");
                    line.className = `terminal-line ${getLevelClass(log.level)}`;
                    line.innerHTML = `[${log.timestamp}] ${log.message}`;
                    terminalContent.appendChild(line);
                });
                
                lastLogCount = data.logs.length;

                if (autoScroll) {
                    terminalWindow.scrollTop = terminalWindow.scrollHeight;
                }
            }
        } catch (err) {
            console.error("Failed to fetch logs", err);
        }
    }

    autoScrollBtn.addEventListener("click", () => {
        autoScroll = !autoScroll;
        if (autoScroll) {
            autoScrollBtn.innerHTML = '<img src="https://win98icons.alexmeub.com/icons/png/scroll-0.png" width="16" height="16"> Auto-Scroll: ON';
            autoScrollBtn.style.boxShadow = "inset 1px 1px var(--win-border-dark), inset -1px -1px var(--win-border-light)";
            terminalWindow.scrollTop = terminalWindow.scrollHeight;
        } else {
            autoScrollBtn.innerHTML = '<img src="https://win98icons.alexmeub.com/icons/png/scroll-0.png" width="16" height="16"> Auto-Scroll: OFF';
            autoScrollBtn.style.boxShadow = "";
        }
    });

    refreshBtn.addEventListener("click", fetchLogs);

    // Poll every 1.5 seconds
    setInterval(fetchLogs, 1500);
    fetchLogs(); // Initial fetch
});
