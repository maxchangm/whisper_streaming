const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const transcription = document.getElementById("transcription");

let mediaRecorder;
let audioChunks = [];
let socket;

// Function to initialize and manage the WebSocket connection
function initializeWebSocket() {
  socket = new WebSocket("ws://localhost:8001/ws");

  socket.onopen = function (event) {
    console.log("WebSocket is open now.");
    transcription.textContent = "Connection established. Start speaking...";
    startButton.disabled = true;
    stopButton.disabled = false;
  };

  socket.onerror = function (error) {
    console.error("WebSocket error observed:", error);
  };

  socket.onmessage = function (event) {
    console.log("Message from server ", event.data);
    transcription.textContent += `\n${event.data}`;
  };

  socket.onclose = function (event) {
    console.log("WebSocket is closed now.", event.reason);
    transcription.textContent = "Connection closed. Please refresh to restart.";
    startButton.disabled = false;
    stopButton.disabled = true;
    setTimeout(initializeWebSocket, 2000); // Try to reconnect after 2 seconds
  };
}

// Ensure the WebSocket connection is open before sending data
function sendData(data) {
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(data);
  } else {
    console.log("WebSocket is not open. ReadyState: ", socket.readyState);
  }
}

// Setup media recorder to capture audio data
function setupMediaRecorder(stream) {
  mediaRecorder = new MediaRecorder(stream);

  mediaRecorder.ondataavailable = function (event) {
    if (event.data.size > 0) {
      audioChunks.push(event.data);
      sendData(event.data);
      audioChunks = []; // Clear the chunk array after sending
    }
  };

  mediaRecorder.onstop = function () {
    socket.close(); // Close the WebSocket when recording stops
  };

  mediaRecorder.start(1000); // Collect 1-second chunks of audio
}

// Handle start button click
startButton.onclick = async function () {
  if (!socket || socket.readyState === WebSocket.CLOSED) {
    initializeWebSocket(); // Initialize WebSocket connection
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
      video: false,
    });
    setupMediaRecorder(stream);
  } catch (error) {
    console.error("Error accessing microphone:", error);
  }
};

// Handle stop button click
stopButton.onclick = function () {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    stopButton.disabled = true;
    startButton.disabled = false;
  }
};

// Initialize WebSocket connection when the page loads
initializeWebSocket();
