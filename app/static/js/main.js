// ========================================
// SIGN LANGUAGE DETECTION - WEB CLIENT
// Mirrors run_inference_multiclass.py functionality
// ========================================

// --- DOM ELEMENTS ---
const video = document.getElementById("webcam");
const landmarkCanvas = document.getElementById("landmarkCanvas");
const landmarkCtx = landmarkCanvas.getContext("2d");
const startButton = document.getElementById("startButton");

// Status & Mode
const modeToggle = document.getElementById("modeToggle");
const modeText = document.getElementById("modeText");
const modeIcon = document.getElementById("modeIcon");
const statusContainer = document.getElementById("statusContainer");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

// Detection Display
const predictionElement = document.getElementById("prediction");
const confidenceElement = document.getElementById("confidence");
const trafficCircle = document.getElementById("trafficCircle");
const trafficInner = document.getElementById("trafficInner");
const trafficLabel = document.getElementById("trafficLabel");

// Progress Bar
const progressContainer = document.getElementById("progressContainer");
const progressBar = document.getElementById("progressBar");
const progressLabel = document.getElementById("progressLabel");

// Spelling & Sentence
const userSaysElement = document.getElementById("userSays");
const spelledWordElement = document.getElementById("spelledWord");
const sentenceElement = document.getElementById("sentence");

// Auto-commit Timer
const autoCommitContainer = document.getElementById("autoCommitContainer");
const autoCommitTimer = document.getElementById("autoCommitTimer");

// Response Panel
const responsePanel = document.getElementById("responsePanel");
const responseText = document.getElementById("responseText");

// Buttons
const finishBtn = document.getElementById("finishBtn");
const localBrainBtn = document.getElementById("localBrainBtn");
const clearBtn = document.getElementById("clearBtn");

// Log
const committedLog = document.getElementById("committedLog");

// --- STATE ---
let isStreaming = false;
let socket = null;
let intervalId = null;
let currentMode = "SPELLING";

// --- CONFIGURATION ---
const FPS = 5;  // Reduced for better performance (5-8 is good balance)
const INTERVAL_MS = 1000 / FPS;
const WS_URL = `ws://${window.location.host}/ws/video`;

// Hand landmark connections for drawing
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],     // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],     // Index
    [0, 9], [9, 10], [10, 11], [11, 12], // Middle
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring
    [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [5, 9], [9, 13], [13, 17]           // Palm
];

// ========================================
// CAMERA & WEBSOCKET
// ========================================

startButton.addEventListener("click", () => {
    if (!isStreaming) {
        startCamera();
    } else {
        stopCamera();
    }
});

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user",
            },
        });
        video.srcObject = stream;
        
        // Wait for video to be ready
        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve();
            };
        });
        
        await video.play();
        
        // Set canvas size to match video dimensions
        // Use a small delay to ensure video is fully rendered
        setTimeout(() => {
            const videoRect = video.getBoundingClientRect();
            landmarkCanvas.width = videoRect.width;
            landmarkCanvas.height = videoRect.height;
            console.log(`Canvas size set to: ${landmarkCanvas.width}x${landmarkCanvas.height}`);
        }, 100);

        isStreaming = true;
        startButton.textContent = "Stop Camera";
        startButton.classList.replace("bg-indigo-600", "bg-red-600");
        startButton.classList.replace("hover:bg-indigo-700", "hover:bg-red-700");

        connectWebSocket();
    } catch (err) {
        console.error("Error accessing webcam:", err);
        updateStatus("ERROR", "red");
    }
}

function stopCamera() {
    isStreaming = false;
    startButton.textContent = "Start Camera";
    startButton.classList.replace("bg-red-600", "bg-indigo-600");
    startButton.classList.replace("hover:bg-red-700", "hover:bg-indigo-700");

    const stream = video.srcObject;
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
    }
    video.srcObject = null;

    if (socket) socket.close();
    if (intervalId) clearInterval(intervalId);
    
    // Clear canvas
    landmarkCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
}

function connectWebSocket() {
    socket = new WebSocket(WS_URL);

    socket.onopen = () => {
        updateStatus("LISTENING", "green");
        console.log("WebSocket connected");
        intervalId = setInterval(sendFrame, INTERVAL_MS);
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateUI(data);
    };

    socket.onclose = () => {
        updateStatus("DISCONNECTED", "red");
        console.log("WebSocket disconnected");
        if (isStreaming) clearInterval(intervalId);
    };

    socket.onerror = (error) => {
        console.error("WebSocket error:", error);
        updateStatus("ERROR", "red");
    };
}

function sendFrame() {
    if (!isStreaming || socket.readyState !== WebSocket.OPEN) return;

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    
    // Reduced resolution for better performance
    const TARGET_WIDTH = 480;
    const TARGET_HEIGHT = 360;
    canvas.width = TARGET_WIDTH;
    canvas.height = TARGET_HEIGHT;

    // Draw frame WITHOUT flip - let backend handle flip
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Increased quality from 0.5 to 0.7 for better accuracy
    canvas.toBlob((blob) => {
        if (blob) socket.send(blob);
    }, "image/jpeg", 0.7);
}

// ========================================
// UI UPDATE FUNCTIONS
// ========================================

function updateUI(data) {
    if (data.error) {
        console.error("Server error:", data.error);
        return;
    }

    // 1. Status Update
    if (data.status) {
        updateStatus(data.status);
    }

    // 2. Mode Update
    if (data.mode) {
        updateModeDisplay(data.mode);
    }

    // 3. Landmarks Visualization
    if (data.landmarks && data.landmarks.length > 0) {
        console.log(`Landmarks received: ${data.landmarks.length} hands, points: ${data.landmarks[0]?.points?.length || 0}`);
        drawLandmarks(data.landmarks);
    } else {
        landmarkCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
    }

    // 4. Prediction & Confidence
    if (data.prediction) {
        predictionElement.textContent = data.prediction;
        confidenceElement.textContent = `${(data.confidence * 100).toFixed(1)}%`;
        updateTrafficLight(data.confidence, data.prediction);
        
        // Animation
        predictionElement.classList.add("scale-110");
        setTimeout(() => predictionElement.classList.remove("scale-110"), 100);
    } else if (data.num_hands === 0) {
        predictionElement.textContent = "-";
        confidenceElement.textContent = "0%";
        updateTrafficLight(0, "");
    }

    // 5. Progress Bar (Reading)
    if (data.stability_progress && data.stability_progress > 0 && data.stability_progress < 1) {
        progressContainer.classList.remove("hidden");
        progressBar.style.width = `${data.stability_progress * 100}%`;
        progressLabel.textContent = `Reading: ${data.potential_label || "..."}`;
    } else {
        progressContainer.classList.add("hidden");
    }

    // 6. Spelling & Sentence
    const spelling = data.spelled_word || data.spelling_update || "-";
    spelledWordElement.textContent = spelling;
    userSaysElement.textContent = data.sentence || spelling || "...";
    
    if (data.sentence !== undefined) {
        sentenceElement.textContent = data.sentence || "";
    }

    // 7. Auto-commit Timer
    if (data.auto_commit_remaining >= 0) {
        autoCommitContainer.classList.remove("hidden");
        autoCommitTimer.textContent = `${Math.ceil(data.auto_commit_remaining)}s`;
        autoCommitTimer.className = data.auto_commit_remaining < 5 
            ? "text-lg font-bold text-red-400" 
            : "text-lg font-bold text-yellow-400";
    } else {
        autoCommitContainer.classList.add("hidden");
    }

    // 8. Response Panel
    if (data.response_text) {
        responsePanel.classList.remove("hidden");
        responseText.textContent = data.response_text;
    }

    // 9. Committed Log
    if (data.committed_log && data.committed_log.length > 0) {
        updateCommittedLog(data.committed_log);
    }

    // 10. Chat Response
    if (data.ai_response || data.bot_response) {
        const response = data.ai_response || data.bot_response;
        responsePanel.classList.remove("hidden");
        responseText.textContent = `${data.ai_response ? "AI" : "Bot"}: ${response}`;
        speakText(response);
    }

    // 11. Audio Action
    if (data.action && data.audio_text) {
        speakText(data.audio_text);
    }
}

function updateStatus(status, color = null) {
    statusText.textContent = status;
    
    const colorMap = {
        "LISTENING": { bg: "bg-green-500/20", border: "border-green-500/30", text: "text-green-400", dot: "bg-green-500" },
        "SPEAKING": { bg: "bg-blue-500/20", border: "border-blue-500/30", text: "text-blue-400", dot: "bg-blue-500" },
        "THINKING": { bg: "bg-purple-500/20", border: "border-purple-500/30", text: "text-purple-400", dot: "bg-purple-500" },
        "DISCONNECTED": { bg: "bg-red-500/20", border: "border-red-500/30", text: "text-red-400", dot: "bg-red-500" },
        "ERROR": { bg: "bg-red-500/20", border: "border-red-500/30", text: "text-red-400", dot: "bg-red-500" }
    };
    
    const colors = colorMap[status] || colorMap["LISTENING"];
    
    statusContainer.className = `flex items-center gap-2 px-4 py-2 rounded-full ${colors.bg} border ${colors.border}`;
    statusText.className = `text-sm font-medium ${colors.text}`;
    statusDot.className = `w-2 h-2 ${colors.dot} rounded-full animate-pulse`;
}

function updateModeDisplay(mode) {
    currentMode = mode;
    modeText.textContent = mode;
    
    if (mode === "WORD") {
        modeToggle.className = "px-4 py-2 bg-green-500 hover:bg-green-600 text-white text-sm font-bold rounded-lg transition-all shadow-md flex items-center gap-2";
        modeIcon.textContent = "📝";
    } else {
        modeToggle.className = "px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-black text-sm font-bold rounded-lg transition-all shadow-md flex items-center gap-2";
        modeIcon.textContent = "✏️";
    }
}

function updateTrafficLight(confidence, label) {
    if (confidence > 0.8) {
        trafficInner.className = "w-6 h-6 rounded-full bg-green-500";
        trafficCircle.className = "w-10 h-10 rounded-full border-4 border-green-500 flex items-center justify-center";
        trafficLabel.textContent = label;
        trafficLabel.className = "text-sm font-bold text-green-400";
    } else if (confidence > 0.5) {
        trafficInner.className = "w-6 h-6 rounded-full bg-yellow-500";
        trafficCircle.className = "w-10 h-10 rounded-full border-4 border-yellow-500 flex items-center justify-center";
        trafficLabel.textContent = `Mungkin: ${label}?`;
        trafficLabel.className = "text-sm font-bold text-yellow-400";
    } else {
        trafficInner.className = "w-6 h-6 rounded-full bg-gray-600";
        trafficCircle.className = "w-10 h-10 rounded-full border-4 border-gray-600 flex items-center justify-center";
        trafficLabel.textContent = "";
    }
}

function updateCommittedLog(logEntries) {
    committedLog.innerHTML = logEntries.map(entry => 
        `<div class="text-green-400">${entry}</div>`
    ).join("");
    committedLog.scrollTop = committedLog.scrollHeight;
}

// ========================================
// LANDMARK DRAWING
// ========================================

function drawLandmarks(landmarksData) {
    // Auto-resize canvas to match video container
    const videoRect = video.getBoundingClientRect();
    if (landmarkCanvas.width !== Math.floor(videoRect.width) || 
        landmarkCanvas.height !== Math.floor(videoRect.height)) {
        landmarkCanvas.width = Math.floor(videoRect.width);
        landmarkCanvas.height = Math.floor(videoRect.height);
    }
    
    landmarkCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
    
    if (!landmarksData || landmarksData.length === 0) return;
    
    const canvasWidth = landmarkCanvas.width;
    const canvasHeight = landmarkCanvas.height;
    
    console.log(`Drawing landmarks on canvas ${canvasWidth}x${canvasHeight}`);
    
    landmarksData.forEach((hand, handIndex) => {
        const points = hand.points;
        if (!points || points.length === 0) return;
        
        const color = handIndex === 0 ? "#00FF00" : "#FF6600"; // Green first, Orange second
        
        // Draw connections first (behind points)
        landmarkCtx.strokeStyle = color;
        landmarkCtx.lineWidth = 3;
        
        HAND_CONNECTIONS.forEach(([start, end]) => {
            if (points[start] && points[end]) {
                const x1 = (points[start].x / 100) * canvasWidth;
                const y1 = (points[start].y / 100) * canvasHeight;
                const x2 = (points[end].x / 100) * canvasWidth;
                const y2 = (points[end].y / 100) * canvasHeight;
                
                landmarkCtx.beginPath();
                landmarkCtx.moveTo(x1, y1);
                landmarkCtx.lineTo(x2, y2);
                landmarkCtx.stroke();
            }
        });
        
        // Draw points
        points.forEach((point, idx) => {
            const x = (point.x / 100) * canvasWidth;
            const y = (point.y / 100) * canvasHeight;
            
            // Larger circle for fingertips (indices 4, 8, 12, 16, 20)
            const radius = [4, 8, 12, 16, 20].includes(idx) ? 8 : 5;
            
            // Draw filled circle
            landmarkCtx.beginPath();
            landmarkCtx.arc(x, y, radius, 0, 2 * Math.PI);
            landmarkCtx.fillStyle = color;
            landmarkCtx.fill();
            
            // White border
            landmarkCtx.strokeStyle = "#FFFFFF";
            landmarkCtx.lineWidth = 2;
            landmarkCtx.stroke();
        });
    });
}

// ========================================
// KEYBOARD SHORTCUTS
// ========================================

document.addEventListener("keydown", (e) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    
    // Prevent default for our shortcuts
    const key = e.key.toLowerCase();
    
    switch (e.key) {
        case "Tab":
            e.preventDefault();
            sendCommand("toggle_mode");
            break;
        case " ": // Space
            e.preventDefault();
            sendCommand("commit_spelling");
            break;
        case "Enter":
            e.preventDefault();
            sendCommand("finish_sentence");
            finishBtn.disabled = true;
            finishBtn.textContent = "Thinking...";
            setTimeout(() => {
                finishBtn.disabled = false;
                finishBtn.textContent = "AI Chat";
            }, 3000);
            break;
        case "8":
            sendCommand("local_brain");
            break;
        case "Backspace":
            e.preventDefault();
            sendCommand("backspace");
            break;
        case "9":
            sendCommand("clear_session");
            break;
    }
});

function sendCommand(command) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(command);
    } else {
        alert("WebSocket not connected. Please start the camera first.");
    }
}

// ========================================
// BUTTON HANDLERS
// ========================================

modeToggle.addEventListener("click", () => sendCommand("toggle_mode"));

finishBtn.addEventListener("click", () => {
    sendCommand("finish_sentence");
    finishBtn.disabled = true;
    finishBtn.textContent = "Thinking...";
    setTimeout(() => {
        finishBtn.disabled = false;
        finishBtn.textContent = "AI Chat";
    }, 3000);
});

localBrainBtn.addEventListener("click", () => sendCommand("local_brain"));

clearBtn.addEventListener("click", () => {
    sendCommand("clear_session");
    // Clear UI instantly
    spelledWordElement.textContent = "-";
    sentenceElement.textContent = "";
    userSaysElement.textContent = "...";
    responsePanel.classList.add("hidden");
    committedLog.innerHTML = '<div class="text-gray-500">Cleared...</div>';
});

// ========================================
// TEXT-TO-SPEECH
// ========================================

function speakText(text) {
    if ("speechSynthesis" in window) {
        // Cancel any ongoing speech
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = "id-ID"; // Indonesian
        utterance.rate = 1.0;
        
        utterance.onend = () => {
            sendCommand("speaking_done");
        };
        
        window.speechSynthesis.speak(utterance);
    } else {
        console.warn("TTS not supported in this browser.");
    }
}

// ========================================
// INITIALIZATION
// ========================================

// Set initial state
updateStatus("READY", "green");
updateModeDisplay("SPELLING");

console.log("Sign Language Detection Web Client loaded");
console.log("Keyboard shortcuts: TAB (mode), SPACE (commit), ENTER (AI), 8 (brain), 9 (clear)");
