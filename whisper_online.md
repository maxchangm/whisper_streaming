# Whisper Online Documentation

`whisper_online.py` is a Python script designed for processing audio in real-time using different Whisper backends for automated speech recognition (ASR). It includes implementations for Whisper-based ASR systems, integration with OpenAI's API, and local Whisper models with both real-time and offline processing capabilities.

## Module Imports

- **sys, io**: For standard input, output, and error stream handling.
- **numpy as np**: For numerical operations on arrays.
- **librosa**: For audio processing.
- **soundfile as sf**: For reading and writing sound files.
- **math, time, logging**: For mathematical functions, timing, and logging operations respectively.
- **functools.lru_cache**: For caching function outputs to optimize performance.

## Functions

### load_audio

- **Purpose**: Loads an audio file and resamples it to 16 kHz.
- **Parameters**: `fname` - file name of the audio.
- **Returns**: Audio data as a numpy array.

### load_audio_chunk

- **Purpose**: Loads a specific chunk from an audio file between specified start and end times.
- **Parameters**:
  - `fname`: Audio file name.
  - `beg`: Beginning time in seconds.
  - `end`: End time in seconds.
- **Returns**: A segment of the audio file as a numpy array.

## Classes

### ASRBase

- **Purpose**: Base class for ASR systems.
- **Methods**:
  - `load_model`: Abstract method to load an ASR model.
  - `transcribe`: Abstract method to transcribe audio.
  - `use_vad`: Abstract method to enable voice activity detection.

### WhisperTimestampedASR

- **Derived from**: ASRBase
- **Purpose**: Implements ASR using the `whisper_timestamped` backend.
- **Methods**:
  - `load_model`: Loads the Whisper model specified.
  - `transcribe`: Transcribes audio using the loaded Whisper model.
  - `ts_words`: Processes transcription results into start, end, and text tuples.

### FasterWhisperASR

- **Derived from**: ASRBase
- **Purpose**: Implements ASR using the `faster-whisper` library.
- **Methods**:
  - `load_model`: Loads the Whisper model with options for GPU acceleration and model size specification.
  - `transcribe`: Performs transcription using Whisper with options for beam size and word timestamps.

### OpenaiApiASR

- **Derived from**: ASRBase
- **Purpose**: Provides ASR functionality using OpenAI's Whisper API.
- **Methods**:
  - `load_model`: Initializes connection to OpenAI's API.
  - `transcribe`: Transcribes audio data using OpenAI's Whisper model, handling different tasks such as translation or transcription.

### HypothesisBuffer

- **Purpose**: Manages a buffer of hypothesized text during real-time transcription.
- **Methods**:
  - `insert`: Inserts new transcribed text into the buffer.
  - `flush`: Commits confirmed segments of text and clears the buffer accordingly.

### OnlineASRProcessor

- **Purpose**: Coordinates the online processing of audio using an ASR model, managing audio buffers and transcription states.
- **Methods**:
  - `insert_audio_chunk`: Inserts new chunks of audio into the processor.
  - `process_iter`: Processes the current buffer and returns confirmed text.
  - `finish`: Finalizes processing and returns any remaining transcribed text.

## Constants

- **WHISPER_LANG_CODES**: Supported language codes for Whisper.

## Helper Functions

- **create_tokenizer**: Creates a tokenizer based on the language code, used for segmenting sentences.
- **add_shared_args**: Adds common command-line arguments for configuring the ASR system.
- **asr_factory**: Factory function to create and configure an ASR instance based on provided arguments.

## Main Execution Logic

- Parses command-line arguments.
- Configures logging.
- Initializes and configures the ASR system.
- Processes audio either in real-time or offline mode based on command-line options.

This script can be adapted for various ASR tasks and configurations, making it versatile for different audio processing needs in real-time environments.
