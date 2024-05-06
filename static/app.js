const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const transcription = document.getElementById('transcription');

let socket = null;
let mediaRecorder = null;
let audioChunks = [];

// Function to start the WebSocket connection
function startWebSocket() {
  if (socket !== null && socket.readyState !== WebSocket.CLOSED) {
    console.log('WebSocket is already open or in the process of opening.');
    return;
  }


    socket = new WebSocket('wss://5dd9-54-237-205-57.ngrok-free.app/ws');

    socket.onopen = function() {
        console.log('WebSocket connection established');
        transcription.textContent = 'Connection established. Start speaking...';
        startButton.disabled = true;
        stopButton.disabled = false;
    };

    socket.onmessage = function(event) {
        console.log('Message from server:', event.data);
        transcription.textContent += `\n${event.data}`;
    };

    socket.onerror = function(event) {
        console.error('WebSocket error observed:', event);
        transcription.textContent = 'WebSocket error. Please check the console.';
    };

    socket.onclose = function(event) {
        console.log('WebSocket is closed now. Reason:', event.reason);
        transcription.textContent = 'Connection closed. Please click start to reconnect.';
        startButton.disabled = false;
        stopButton.disabled = true;
        socket = null;
    };
}

// Function to stop the WebSocket connection
function stopWebSocket() {
    if (socket) {
        socket.close();
        socket = null;
        console.log('WebSocket connection closed by client');
    }
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
    }
    startButton.disabled = false;
    stopButton.disabled = true;
}

// Setup media recorder to capture audio data
async function setupMediaRecorder() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = function(event) {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
                if (socket && socket.readyState === WebSocket.OPEN) {
                    socket.send(event.data);
                }
                audioChunks = [];  // Clear the chunk array after sending
            }
        };

        mediaRecorder.onstop = function() {
            stopWebSocket();
        };

        mediaRecorder.start(1000);  // Collect 1-second chunks of audio
    } catch (error) {
        console.error('Error accessing microphone:', error);
        transcription.textContent = 'Could not access microphone. Please check permissions.';
    }
}

// Event listeners for buttons
startButton.addEventListener('click', function() {
    startWebSocket();
    setupMediaRecorder();
});

stopButton.addEventListener('click', function() {
    stopWebSocket();
});
