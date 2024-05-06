#!/usr/bin/env python3
from whisper_online import *

import sys
import argparse
import os
import logging
import numpy as np
from fastapi import FastAPI, WebSocket

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

SAMPLING_RATE = 16000

asr = FasterWhisperASR(lan="yue", modelsize="large-v2")
online = OnlineASRProcessor(asr)
min_chunk = 1



######### Server objects

import line_packet
import socket

#  comments for Connection class
#  it wraps conn object
#  PACKET_SIZE is the size of the packet that is sent to the client
#  send method sends the line to the client, but it doesn't send the same line twice, because it was problematic in online-text-flow-events
#  receive_lines method receives the lines from the client
#  non_blocking_receive_audio method receives the audio from the client


class Connection:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def send(self, message: str):
        """Sends a text message to the client over the WebSocket connection.

        Args:
            message (str): The message to be sent.
        """
        await self.websocket.send_text(message)

    async def receive_audio_chunk(self):
        """Receives audio data as bytes from the WebSocket connection.

        Returns:
            bytes: The audio data received from the client, or None if an error occurs.
        """
        try:
            # This method will wait for a message from the client and return the data as bytes.
            data = await self.websocket.receive_bytes()
            return data
        except Exception as e:
            # If any exception occurs during reception, log or print the exception message and return None.
            print(f"Error receiving data: {e}")
            return None


import io
import soundfile


# wraps socket and ASR object, and serves one client connection.
# next client should be served by a new instance of this object
class ServerProcessor:
    def __init__(self, connection, online_asr_proc, min_chunk_size):
        self.connection = connection
        self.online_asr_proc = online_asr_proc
        self.min_chunk_size = min_chunk_size
        self.last_end = None  # Used to ensure non-overlapping intervals for transcript segments

    async def process(self):
        """Process the audio stream using the ASR model and send results."""
        self.online_asr_proc.init()
        try:
            while True:
                audio = await self.connection.receive_audio_chunk()
                if audio is None:  # If no audio is received, assume the connection is closed or there's an error
                    break
                self.online_asr_proc.insert_audio_chunk(audio)
                result = self.online_asr_proc.process_iter()
                formatted_transcript = self.format_output_transcript(result)
                await self.send_result(formatted_transcript)
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            print("Processing completed.")

    def format_output_transcript(self, transcript_data):
        """Formats the output transcript to ensure proper display or further processing."""
        if transcript_data and transcript_data[2]:  # Check if there is actual text to format
            beg, end, text = transcript_data
            if self.last_end is not None:
                beg = max(beg, self.last_end)
            self.last_end = end
            return f"{beg:.3f} {end:.3f} {text}"
        else:
            return None

    async def send_result(self, formatted_transcript):
        """Sends the formatted transcript result to the client via WebSocket."""
        if formatted_transcript:
            await self.connection.send(formatted_transcript)


#        o = online.finish()  # this should be working
#        self.send_result(o)


# server loop

# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.listen(1)
#     while True:
#         conn, addr = s.accept()
#         logger.info("Connected to client on {}".format(addr))
#         connection = Connection(conn)
#         proc = ServerProcessor(connection, online, min_chunk)
#         proc.process()
#         conn.close()
#         logger.info("Connection to client closed")
# logger.info("Connection closed, terminating.")
