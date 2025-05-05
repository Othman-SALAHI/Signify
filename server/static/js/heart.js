document.addEventListener("DOMContentLoaded", () => {
    const cameraFeed = document.getElementById("camera-feed");
    const cameraPlaceholder = document.getElementById("camera-placeholder");
    const toggleCameraBtn = document.getElementById("toggle-camera");
    const toggleProcessingBtn = document.getElementById("toggle-processing");
    const clearWordBtn = document.getElementById("clear-word");
    const deleteLetterBtn = document.getElementById("delete-letter");
    const textToSpeechBtn = document.getElementById("text-to-speech");
    const predictedLetter = document.getElementById("predicted-letter");
    const currentWord = document.getElementById("current-word");
    const confidenceBar = document.getElementById("confidence-bar");
    const confidenceValue = document.getElementById("confidence-value");
    const handCanvas = document.getElementById("hand-canvas");
    const handCtx = handCanvas.getContext("2d");
    const canvasPlaceholder = document.getElementById("canvas-placeholder");
  
    let isCameraOn = false;
    let isProcessingOn = false;
    let predictionInterval = null;
  
    function resizeCanvas() {
      handCanvas.width = handCanvas.offsetWidth;
      handCanvas.height = handCanvas.offsetHeight;
    }
  
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
  
    async function startCamera() {
      toggleCameraBtn.disabled = true;
      toggleCameraBtn.textContent = "Starting...";
      await fetch("/start_camera", { method: "POST" });
      cameraPlaceholder.style.display = "none";
      cameraFeed.src = "/video_feed?" + Date.now(); // prevent cache
      cameraFeed.style.display = "block";
      toggleCameraBtn.textContent = "Stop Camera";
      toggleProcessingBtn.disabled = false;
      isCameraOn = true;
      toggleCameraBtn.disabled = false;
    }
  
    async function stopCamera() {
      await fetch("/stop_camera", { method: "POST" });
      cameraFeed.src = "";
      cameraFeed.style.display = "none";
      cameraPlaceholder.style.display = "flex";
      toggleCameraBtn.textContent = "Start Camera";
      toggleProcessingBtn.disabled = true;
      isCameraOn = false;
      stopPrediction();
    }
  
    function startPrediction() {
      isProcessingOn = true;
      toggleProcessingBtn.textContent = "Stop Processing";
      canvasPlaceholder.style.display = "none";
  
      predictionInterval = setInterval(async () => {
        const res = await fetch("/get_phrase");
        const data = await res.json();
  
        predictedLetter.textContent = data.letter || "[ ]";
        currentWord.textContent = data.phrase || "[ ]";
        const confidence = data.confidence || 0;
        confidenceBar.style.width = `${Math.round(confidence * 100)}%`;
        confidenceValue.textContent = `${Math.round(confidence * 100)}%`;
  
        drawLandmarks(data.landmarks);
      }, 1000);
    }
  
    function stopPrediction() {
      isProcessingOn = false;
      clearInterval(predictionInterval);
      toggleProcessingBtn.textContent = "Start Processing";
      canvasPlaceholder.style.display = "flex";
      handCtx.clearRect(0, 0, handCanvas.width, handCanvas.height);
    }
  
    function drawLandmarks(landmarks) {
      handCtx.clearRect(0, 0, handCanvas.width, handCanvas.height);
      if (!landmarks || landmarks.length !== 63) return;
  
      handCtx.strokeStyle = "rgba(79,70,229,0.7)";
      handCtx.fillStyle = "rgba(79,70,229,0.9)";
      handCtx.lineWidth = 2;
  
      const connections = [
        [0,1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20],[0,5,9,13,17]
      ];
  
      connections.forEach(conn => {
        handCtx.beginPath();
        conn.forEach((idx, i) => {
          const x = landmarks[idx * 3] * handCanvas.width;
          const y = landmarks[idx * 3 + 1] * handCanvas.height;
          i === 0 ? handCtx.moveTo(x, y) : handCtx.lineTo(x, y);
        });
        handCtx.stroke();
      });
  
      for (let i = 0; i < 21; i++) {
        const x = landmarks[i * 3] * handCanvas.width;
        const y = landmarks[i * 3 + 1] * handCanvas.height;
        handCtx.beginPath();
        handCtx.arc(x, y, 3, 0, Math.PI * 2);
        handCtx.fill();
      }
    }
  
    toggleCameraBtn.addEventListener("click", () => {
      isCameraOn ? stopCamera() : startCamera();
    });
  
    toggleProcessingBtn.addEventListener("click", () => {
      isProcessingOn ? stopPrediction() : startPrediction();
    });
  
    clearWordBtn.addEventListener("click", async () => {
      await fetch("/clear_phrase", { method: "POST" });
      currentWord.textContent = "[ ]";
      predictedLetter.textContent = "[ ]";
    });

    deleteLetterBtn.addEventListener("click", async () => {
      await fetch("/delete_letter", { method: "POST" });
      // Re-fetch phrase from server
      const res = await fetch("/get_phrase");
      const data = await res.json();
      currentWord.textContent = data.phrase || "[ ]";
    });
    textToSpeechBtn.addEventListener("click", async () => {
      const res = await fetch("/text_to_speech", { method: "POST" });
      const blob = await res.blob();
      const audio = new Audio(URL.createObjectURL(blob));
      audio.play();
    });
  
    stopCamera(); // initialize off
  });
  