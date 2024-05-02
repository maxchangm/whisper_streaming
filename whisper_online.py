#!/usr/bin/env python3
import sys
import numpy as np
import librosa
from functools import lru_cache
import time
import logging


import io
import soundfile as sf
import math

logger = logging.getLogger(__name__)


@lru_cache
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a


def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg * 16000)
    end_s = int(end * 16000)
    return audio[beg_s:end_s]


# Whisper backend


class ASRBase:

    sep = " "  # join transcribe words with this character (" " for whisper_timestamped,
    # "" for faster-whisper because it emits the spaces when neeeded)

    def __init__(
        self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr
    ):
        self.logfile = logfile

        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan

        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize, cache_dir):
        raise NotImplemented("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplemented("must be implemented in the child class")

    def use_vad(self):
        raise NotImplemented("must be implemented in the child class")


# write comments for FasterWhisperASR class
# it is a subclass of ASRBase class
# it uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version.
class FasterWhisperASR(ASRBase):
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version."""

    sep = ""

    # it is a method of ASRBase class
    # it loads the model from the model_dir or model_cache_dir
    # it uses the model_size parameter to load the model
    # it returns the model object
    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        #        logging.getLogger("faster_whisper").setLevel(logger.level)
        if model_dir is not None:
            logger.debug(
                f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used."
            )
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")

        # this worked fast and reliably on NVIDIA L40
        model = WhisperModel(
            model_size_or_path,
            device="cuda",
            compute_type="float16",
            download_root=cache_dir,
        )

        # or run on GPU with INT8
        # tested: the transcripts were different, probably worse than with FP16, and it was slightly (appx 20%) slower
        # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

        # or run on CPU with INT8
        # tested: works, but slow, appx 10-times than cuda FP16
        #        model = WhisperModel(modelsize, device="cpu", compute_type="int8") #, download_root="faster-disk-cache-dir/")
        return model

    # it is a method of ASRBase class
    # it uses the model object to transcribe the audio
    # it returns the transcription result
    # it uses the init_prompt parameter to initialize the prompt
    # it uses the vad parameter to use VAD
    # it uses the buffer_trimming parameter to trim the completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.
    # it uses the buffer_trimming_sec parameter to trim the completed segments longer than s, and it is used only for "sentence" option
    # it returns the transcription result
    def transcribe(self, audio, init_prompt=""):

        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        # print(info)  # info contains language detection result

        return list(segments)

    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


"""
The HypothesisBuffer serves as a specialized buffer that manages transcription segments or hypotheses during the ASR process. 
It temporarily stores these hypotheses, checks them for consistency and accuracy against new incoming data, 
and commits them to a final output once they are confirmed.
"""
class HypothesisBuffer:

    def __init__(self, logfile=sys.stderr):

        # The variables together help manage the dynamic data of transcribed audio, ensuring that the ASR process is efficient,
        # accurate, and capable of handling real-time audio streams effectively. They allow the buffer to dynamically adjust
        # and commit data based on ongoing analyses and comparisons, contributing crucially to the accuracy and reliability of the ASR output.

        # List to store segments that have been fully processed and confirmed as accurate.
        self.commited_in_buffer = []

        # Temporary storage for segments that are currently being processed but not yet confirmed.
        self.buffer = []

        # Stores new transcription segments that are pending processing to check against the buffer for confirmation.
        self.new = []

        # Tracks the timestamp of the last committed transcription to help manage the buffer's state and ensure continuity.
        self.last_commited_time = 0

        # Stores the last committed word for reference, which can be used to optimize processing or resolve ambiguities in speech recognition.
        self.last_commited_word = None

        # Log file for debugging and tracking the internal state and decisions of the buffer management.
        self.logfile = logfile

    def insert(self, new, offset):
        # Adjust timestamps of new words by the provided offset and update the new buffer
        # For each tuple in the list 'new', where each tuple consists of a start time 'a', an end time 'b', and a transcript 't',
        # create a new tuple where the start and end times are increased by the offset value, and keep the transcript 't' unchanged.
        # Collect all these new tuples into a new list named 'new'.
        new = [
            (segment_start + offset, segment_end + offset, t)
            for segment_start, segment_end, t in new
        ]

        # Update the attribute self.new to only include entries from the modified 'new' list where the start time 'segment_start' is greater than the last committed time minus 0.1 seconds.
        # This ensures that the buffer only contains new words not yet committed, preventing redundant processing of words that have already been considered in the transcription.
        # Filtering out words that occur too close to or before the last committed time helps maintain the integrity of the transcription process by avoiding duplicate entries.# Filter new entries to only include words that occur after the last committed word's time
        # Update the attribute self.new to a new list that contains only the tuples (segment_start, segment_end, t) from the modified new list where the start time segment_start of each tuple is greater than last_commited_time minus 0.1 seconds.
        self.new = [
            (segment_start, segment_end, t)
            for segment_start, segment_end, t in new
            if segment_start > self.last_commited_time - 0.1
        ]

        # If there is at least one new word close to the last committed time, check for overlaps
        if len(self.new) >= 1:
            segment_start, segment_end, t = self.new[0]
            # Check if the starting time of the new word is close to the last committed time
            if abs(segment_start - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # Look for overlapping n-grams between new and committed buffers, up to a maximum of 5 words
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(
                        1, min(min(cn, nn), 5) + 1
                    ):  # Check overlapping n-grams of increasing length
                        c = " ".join(
                            [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][
                                ::-1
                            ]
                        )
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        # If a matching sequence is found, remove the redundant words from `self.new`
                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = " ".join(words)
                            logger.debug(f"removing last {i} words: {words_msg}")
                            break

    def flush(self):
        """
        Processes the buffer to commit the longest common prefix of the two most recent inserts.
        This method finalizes and commits segments of transcription where there is agreement between 'new' and 'buffer'.

        Why it's needed:
        - Ensures transcription accuracy by committing only those segments that are confirmed by subsequent data.
        - Helps maintain a clean buffer by removing processed segments, which contributes to managing memory and improving processing speed.

        How it contributes:
        - By confirming and committing data, it aids in producing a final transcription output that is accurate and reliable.
        - Directly influences the overall ASR process by integrating real-time data validation and refinement of transcription hypotheses.

        Returns:
        - A list of committed transcription segments.
        """
        committed_segments = []
        # List to hold segments that are confirmed and ready to be committed
        # Iterate through the new buffer to check each segment against the main buffer
        while self.new:
            start_time, end_time, text = self.new[0]  # Destructure the first segment tuple from new segments
            if len(self.buffer) == 0:
                break  # Exit if there are no segments in the main buffer to compare against

            # Check if the text from the new segment matches the text from the first segment in the main buffer
            # If there is a match, add the segment to the list of committed segments and remove it from the main buffer
            if text == self.buffer[0][2]:
                committed_segments.append((start_time, end_time, text))  # Add the matching segment to the list of committed segments
                logger.info("Committed segment: " + committed_segments)
                self.last_commited_word = text  # Update the last committed word to the current text
                

                self.last_commited_time = end_time  # Update the last committed time to the end time of the current segment
                self.buffer.pop(0)  # Remove the matched segment from the main buffer
                self.new.pop(0)  # Remove the processed segment from the new buffer
            else:
                break  # Stop processing if there is no match

            self.buffer = self.new
            # Reset the main buffer to include only unprocessed new segments
            self.new = []  # Clear the new segments buffer
            self.commited_in_buffer.extend(committed_segments)
            # Extend the committed in buffer with the newly committed segments
            return committed_segments  # Return the list of newly committed segments

    def pop_commited(self, time):
        """
        Removes entries from the committed buffer that are older or equal to the specified time.

        Why it's needed:
        - Helps manage memory by clearing out old data that is no longer needed.
        - Prevents outdated data from interfering with new data processing, which is crucial for maintaining transcription accuracy.

        How it contributes:
        - Supports the real-time processing aspect of the ASR system by keeping the buffer updated and relevant to the current audio processing context.
        """
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        """
        Returns the current buffer without modifying it.

        Why it's needed:
        - Allows for retrieval of the current state of transcription without committing it, useful for checks and intermediate processing steps.

        How it contributes:
        - Provides a mechanism to access the current transcription hypotheses without affecting the buffer's state, essential for debugging and real-time updates in the ASR process.

        Returns:
        - The current buffer as a list of transcription segments.
        """
        return self.buffer


"""
This class is responsible for managing the overall process of converting audio input into textual transcription. 
It handles audio input, segments the audio for processing, 
utilizes an automatic speech recognition (ASR) model to generate transcription hypotheses, 
and manages these hypotheses through various stages of confirmation.

The OnlineASRProcessor uses the HypothesisBuffer to store and manage unconfirmed text segments. 
As new audio data is processed and transcribed, the OnlineASRProcessor feeds these new transcription results into the HypothesisBuffer. 
The buffer then evaluates these new inputs, confirming them against existing data,
and commits confirmed segments to the final transcription output. 
This interaction ensures a dynamic and efficient handling of the transcription process, 
where only validated text is outputted, minimizing errors and inaccuracies in real-time transcription.
"""
class OnlineASRProcessor:

    SAMPLING_RATE = 16000

    def __init__(
        self, asr, tokenizer=None, buffer_trimming=("segment", 15), logfile=sys.stderr
    ):
        """asr: WhisperASR object
        tokenizer: sentence tokenizer object for the target language. Must have a method *split* that behaves like the one of MosesTokenizer. It can be None, if "segment" buffer trimming option is used, then tokenizer is not used at all.
        ("segment", 15)
        buffer_trimming: a pair of (option, seconds), where option is either "sentence" or "segment", and seconds is a number. Buffer is trimmed if it is longer than "seconds" threshold. Default is the most recommended option.
        logfile: where to store the log.
        """
        self.asr = asr
        self.tokenizer = tokenizer
        self.logfile = logfile

        self.init()

        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

    def init(self):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0

        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.commited = []

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of commited text that is inside of the scrolled away part of audio buffer.
        "context" is the commited text that is inside the audio buffer. It is transcribed again and skipped. It is returned only for debugging and logging reasons.
        """
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1

        p = self.commited[:k]
        p = [t for _, _, t in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x) + 1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(
            t for _, _, t in non_prompt
        )

    # it is a method that returns a tuple (prompt, context), where "prompt" is a 200-character suffix of commited text that is inside of the scrolled away part of audio buffer.
    # "context" is the commited text that is inside the audio buffer. It is transcribed again and skipped. It is returned only for debugging and logging reasons.
    # it is used by the prompt method
    def process_iter(self):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, "").
        The non-emty text is confirmed (committed) partial transcript.

        Processes the current audio buffer and handles the transcription process,
        managing the cycle of confirming or rejecting transcription hypotheses.

        - Retrieves a prompt and its context from the transcript history to maintain continuity.
        - Transcribes the current audio buffer using the ASR model.
        - Converts raw ASR output into structured data with timestamps.
        - Inserts these data into the hypothesis buffer for comparison and potential commitment.
        - Retrieves and logs any uncommitted text, which is crucial for debugging and understanding the parts of the audio not yet finalized.
        - Manages the buffer by committing confirmed transcripts and reporting the status of transcription, which includes completed and in-process text.
        - Adjusts the audio buffer based on trimming settings to optimize memory usage and processing time.

        This method is central to the operation of the ASR system, ensuring that audio is processed efficiently,
        transcription accuracy is maintained, and resources are managed effectively. It contributes to the ASR process by
        dynamically managing audio and text data, ensuring that only accurate, confirmed transcriptions are forwarded
        to the next stages of the application or stored.
        """

        """ Generate a prompt based on previously confirmed text to maintain contextual relevance in transcription. """
        prompt, non_prompt = self.prompt()
        logger.debug(f"PROMPT: {prompt}")
        logger.debug(f"CONTEXT: {non_prompt}")
        logger.debug(f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}")

        """Transcribe the current audio buffer using the provided ASR model with the generated prompt for context."""
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)

        """ # Convert the raw transcription results into structured data with timestamps for each word. 
        Transform to [(beg,end,"word1"), ...] """
        tsw = self.asr.ts_words(res)

        """ Insert these timestamped words into the hypothesis buffer to check against previously stored data. """
        self.transcript_buffer.insert(tsw, self.buffer_time_offset)

        """ Attempt to commit words from the hypothesis buffer that are confirmed by new input."""
        committedSegments = self.transcript_buffer.flush()
        if committedSegments is None:
            committedSegments = []
        self.commited.extend(committedSegments)

        logger.info(f"Current commited text: {[(t[2], t[0], t[1]) for t in self.commited]}")

        unconfirmed_text = (
            self.transcript_buffer.complete()
        )  # Retrieves uncommitted text from the buffer
        if unconfirmed_text:
            logger.debug(
                f"UNCONFIRMED TEXT: {[(t[2], t[0], t[1]) for t in unconfirmed_text]}"
            )

        """ Prepare the committed text for output by combining timestamped segments into a single formatted string. """
        completed = self.to_flush(committedSegments)
        logger.debug(f">>>>COMPLETE NOW: {completed}")

        """ Also prepare the remaining unconfirmed text for possible debugging or further processing."""
        the_rest = self.to_flush(self.transcript_buffer.complete())
        logger.debug(f"INCOMPLETE: {the_rest}")

        # there is a newly confirmed text
        """ Manage the audio buffer based on the specified trimming strategy to optimize memory usage and performance."""
        if (
            committedSegments and self.buffer_trimming_way == "sentence"
        ):  # trim the completed sentences
            if (
                len(self.audio_buffer) / self.SAMPLING_RATE > self.buffer_trimming_sec
            ):  # longer than this
                self.chunk_completed_sentence()

        if self.buffer_trimming_way == "segment":
            s = self.buffer_trimming_sec  # trim the completed segments longer than s,
        else:
            s = 30  # if the audio buffer is longer than 30s, trim it

        """Trim the buffer if it exceeds the specified threshold, ensuring efficient processing."""
        if len(self.audio_buffer) / self.SAMPLING_RATE > s:
            self.chunk_completed_segment(res)

            # alternative: on any word
            # l = self.buffer_time_offset + len(self.audio_buffer)/self.SAMPLING_RATE - 10
            # let's find commited word that is less
            # k = len(self.commited)-1
            # while k>0 and self.commited[k][1] > l:
            #    k -= 1
            # t = self.commited[k][1]
            logger.debug("chunking segment")
            # self.chunk_at(t)

        logger.debug(
            f"len of buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}"
        )

        """ Return the formatted, confirmed text for this iteration."""
        return self.to_flush(committedSegments)

    def chunk_completed_sentence(self):
        if self.commited == []:
            return
        logger.debug(self.commited)
        sents = self.words_to_sentences(self.commited)
        for s in sents:
            logger.debug(f"\t\tSENT: {s}")
        if len(sents) < 2:
            return
        while len(sents) > 2:
            sents.pop(0)
        # we will continue with audio processing at this timestamp
        chunk_at = sents[-2][1]

        logger.debug(f"--- sentence chunked at {chunk_at:2.2f}")
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        if self.commited == []:
            return

        ends = self.asr.segments_end_ts(res)

        t = self.commited[-1][1]

        if len(ends) > 1:

            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= t:
                logger.debug(f"--- segment chunked at {e:2.2f}")
                self.chunk_at(e)
            else:
                logger.debug(f"--- last segment not within commited area")
        else:
            logger.debug(f"--- not enough segments to chunk")

    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time" """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * self.SAMPLING_RATE) :]
        self.buffer_time_offset = time

    def words_to_sentences(self, words):
        """Uses self.tokenizer for sentence segmentation of words.
        Returns: [(beg,end,"sentence 1"),...]
        """

        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        s = self.tokenizer.split(t)
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b, e, w = cwords.pop(0)
                w = w.strip()
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg, end, fsent))
                    break
                sent = sent[len(w) :].strip()
        return out

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logger.debug("last, noncommited: {f}")
        return f

    def to_flush(
        self,
        sents,
        sep=None,
        offset=0,
    ):
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b, e, t)


WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(
    ","
)


def create_tokenizer(lan):
    """returns an object that has split function that works like the one of MosesTokenizer"""

    assert (
        lan in WHISPER_LANG_CODES
    ), "language must be Whisper's supported lang code: " + " ".join(WHISPER_LANG_CODES)

    if lan == "uk":
        import tokenize_uk

        class UkrainianTokenizer:
            def split(self, text):
                return tokenize_uk.tokenize_sents(text)

        return UkrainianTokenizer()

    # supported by fast-mosestokenizer
    if (
        lan
        in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh".split()
    ):
        from mosestokenizer import MosesTokenizer

        return MosesTokenizer(lan)

    # the following languages are in Whisper, but not in wtpsplit:
    if (
        lan
        in "as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt".split()
    ):
        logger.debug(
            f"{lan} code is not supported by wtpsplit. Going to use None lang_code option."
        )
        lan = None

    from wtpsplit import WtP

    # downloads the model from huggingface on the first use
    wtp = WtP("wtp-canine-s-12l-no-adapters")

    class WtPtok:
        def split(self, sent):
            return wtp.split(sent, lang_code=lan)

    return WtPtok()


def add_shared_args(parser):
    """shared args for simulation (this entry point) and server
    parser: argparse.ArgumentParser object
    """
    parser.add_argument(
        "--min-chunk-size",
        type=float,
        default=1.0,
        help="Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v2",
        choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large".split(
            ","
        ),
        help="Name size of the Whisper model to use (default: large-v2). The model is automatically downloaded from the model hub if not present in model cache dir.",
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=None,
        help="Overriding the default model cache dir where models downloaded from the hub are saved",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.",
    )
    parser.add_argument(
        "--lan",
        "--language",
        type=str,
        default="auto",
        help="Source language code, e.g. en,de,cs, or 'auto' for language detection.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Transcribe or translate.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="faster-whisper",
        choices=["faster-whisper", "whisper_timestamped", "openai-api"],
        help="Load only this backend for Whisper processing.",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        default=False,
        help="Use VAD = voice activity detection, with the default parameters.",
    )
    parser.add_argument(
        "--buffer_trimming",
        type=str,
        default="segment",
        choices=["sentence", "segment"],
        help='Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.',
    )
    parser.add_argument(
        "--buffer_trimming_sec",
        type=float,
        default=15,
        help="Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the log level",
        default="DEBUG",
    )


def asr_factory(args, logfile=sys.stderr):
    """
    Creates and configures an ASR and ASR Online instance based on the specified backend and arguments.
    """
    backend = args.backend

    if backend == "faster-whisper":
        asr_cls = FasterWhisperASR
    else:
        raise Exception("Can only be faster whisper backend")

    # Only for FasterWhisperASR and WhisperTimestampedASR
    size = args.model
    t = time.time()
    logger.info(f"Loading Whisper {size} model for {args.lan}...")
    asr = asr_cls(
        modelsize=size,
        lan=args.lan,
        cache_dir=args.model_cache_dir,
        model_dir=args.model_dir,
    )
    e = time.time()
    logger.info(f"done. It took {round(e-t,2)} seconds.")

    # Apply common configurations
    if getattr(args, "vad", False):  # Checks if VAD argument is present and True
        logger.info("Setting VAD filter")
        asr.use_vad()

    language = args.lan
    if args.task == "translate":
        asr.set_translate_task()
        tgt_language = "en"  # Whisper translates into English
    else:
        tgt_language = language  # Whisper transcribes in this language

    # Create the tokenizer
    if args.buffer_trimming == "sentence":
        tokenizer = create_tokenizer(tgt_language)
    else:
        tokenizer = None

    # Create the OnlineASRProcessor
    online = OnlineASRProcessor(
        asr,
        tokenizer,
        logfile=logfile,
        buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec),
    )

    return asr, online


def set_logging(args, logger, other="_server"):
    logging.basicConfig(format="%(levelname)s\t%(message)s")  # format='%(name)s
    logger.setLevel(args.log_level)
    logging.getLogger("whisper_online" + other).setLevel(args.log_level)


#    logging.getLogger("whisper_online_server").setLevel(args.log_level)


if __name__ == "__main__":

    # Step 1: Command Line Argument Parsing
    # The script starts by defining an argument parser using `argparse.ArgumentParser`
    # and setting up necessary command-line arguments such as `--model`, `--language`, etc.
    # The provided arguments (`input_audio.wav`, `--model large-v3`, etc.) are parsed.
    # These specify the configuration like the model size, language, backend, and various operational parameters.

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_path",
        type=str,
        help="Filename of 16kHz mono channel wav, on which live streaming is simulated.",
    )
    add_shared_args(parser)
    parser.add_argument(
        "--start_at",
        type=float,
        default=0.0,
        help="Start processing audio at this time.",
    )
    parser.add_argument(
        "--offline", action="store_true", default=False, help="Offline mode."
    )
    parser.add_argument(
        "--comp_unaware",
        action="store_true",
        default=False,
        help="Computationally unaware simulation.",
    )

    args = parser.parse_args()

    # reset to store stderr to different file stream, e.g. open(os.devnull,"w")
    logfile = sys.stderr

    if args.offline and args.comp_unaware:
        logger.error(
            "No or one option from --offline and --comp_unaware are available, not both. Exiting."
        )
        sys.exit(1)

    #    if args.log_level:
    #        logging.basicConfig(format='whisper-%(levelname)s:%(name)s: %(message)s',
    #                            level=getattr(logging, args.log_level))

    # Step 2: Logging Configuration
    # The `set_logging` function is called to configure the logging based on the specified log level (defaults to `DEBUG` if not provided).
    # This setup controls the output of log messages throughout the script.
    set_logging(args, logger)

    audio_path = args.audio_path

    SAMPLING_RATE = 16000
    duration = len(load_audio(audio_path)) / SAMPLING_RATE
    logger.info("Audio duration is: %2.2f seconds" % duration)

    # Step 3: ASR System Initialization
    # The `asr_factory` function is invoked with parsed arguments.
    # This function decides which ASR backend to initialize based on the `--backend` argument.
    # In your case, since `--backend faster-whisper` is specified, an instance of `FasterWhisperASR` is created.
    # The `FasterWhisperASR` class initializes the Whisper model with the specified size (`large-v3`)
    # and language (`yue`), along with setting up any specified options like voice activity detection (`--vad`).
    # Step 4: Online ASR Processor Setup (inside asr_factory)
    # An instance of `OnlineASRProcessor` is created using the initialized ASR model.
    # The `OnlineASRProcessor` handles real-time audio processing, manages buffers, and coordinates the transcription process.
    # Buffer trimming strategy and threshold are configured (`--buffer_trimming segment` and `--buffer_trimming_sec 2`).
    asr, online = asr_factory(args, logfile=logfile)
    min_chunk = args.min_chunk_size

    # load the audio into the LRU cache before we start the timer
    # Step 5: Audio Loading and Processing
    # The `load_audio` function loads the specified audio file (`input_audio.wav`) into memory.
    # This function uses `librosa` to load the audio at a sampling rate of 16,000 Hz, which is suitable for the Whisper model.
    # The audio is processed in chunks, with each chunk size determined by the `--min-chunk-size` parameter (1 second in your case).
    # This means the script processes the audio in one-second intervals.
    a = load_audio_chunk(audio_path, 0, 1)

    # warm up the ASR because the very first transcribe takes much more time than the other
    # Step 6: Transcription Loop
    # For each audio chunk, the `OnlineASRProcessor` performs several operations:
    #     - **Insert Audio Chunk**: The current audio chunk is added to the processor’s audio buffer.
    #     - **Transcription Process**: The `process_iter` method is called to transcribe the current audio buffer, using the Whisper model.
    # This method handles the transcription, manages hypothesis buffers, and commits confirmed transcription segments.
    #     - **Output Transcription**: As transcription results are confirmed, they are output to the standard output or log, based on the logging configuration.
    asr.transcribe(a)

    beg = args.start_at
    start = time.time() - beg
    logger.info("Start time is: %2.2f seconds" % start)

    def output_transcript(o, now=None):
        # output format in stdout is like:
        # 4186.3606 0 1720 Takhle to je
        # - the first three words are:
        #    - emission time from beginning of processing, in milliseconds
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if now is None:
            now = time.time() - start
        if o[0] is not None:
            print(
                "%1.4f %1.0f %1.0f %s" % (now * 1000, o[0] * 1000, o[1] * 1000, o[2]),
                file=logfile,
                flush=True,
            )
            print(
                "%1.4f %1.0f %1.0f %s" % (now * 1000, o[0] * 1000, o[1] * 1000, o[2]),
                flush=True,
            )
        else:
            # No text, so no output
            pass

    # Step 7: Buffer Management
    # Depending on the buffer trimming configuration, the audio and transcription buffers are managed to avoid overgrowth and to optimize memory usage.
    # In your setup, the buffer is trimmed based on completed segments longer than 2 seconds.
    if args.offline:  ## offline mode processing (for testing/debugging)
        a = load_audio(audio_path)
        online.insert_audio_chunk(a)
        try:
            o = online.process_iter()
        except AssertionError as e:
            log.error(f"assertion error: {repr(e)}")
        else:
            output_transcript(o)
        now = None
    elif args.comp_unaware:  # computational unaware mode
        end = beg + min_chunk
        while True:
            a = load_audio_chunk(audio_path, beg, end)
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {repr(e)}")
                pass
            else:
                output_transcript(o, now=end)

            logger.debug(f"## last processed {end:.2f}s")

            if end >= duration:
                break

            beg = end

            if end + min_chunk > duration:
                end = duration
            else:
                end += min_chunk
        now = duration

    else:  # online = simultaneous mode
        end = 0
        while True:
            now = time.time() - start
            logger.info("Now time is: %2.2f seconds" % now)
            logger.info("End time is: %2.2f seconds" % end)

            if now < end + min_chunk:
                logger.info("Sleeping for %2.2f seconds" % (min_chunk + end - now))
                time.sleep(min_chunk + end - now)
            end = time.time() - start
            logger.info("End time is: %2.2f seconds" % end)
            a = load_audio_chunk(audio_path, beg, end)
            beg = end
            logger.info("Beg time is now end time: %2.2f seconds" % beg)
            online.insert_audio_chunk(a)

            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {e}")
                pass
            else:
                output_transcript(o)
            now = time.time() - start
            logger.debug(
                f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}"
            )

            if end >= duration:
                break
        now = None

    # Step 8: Completion
    # Once all audio has been processed, any remaining unconfirmed or partially processed transcription data is handled by the `finish` method of `OnlineASRProcessor`,
    # ensuring that no transcription data is lost at the end of the processing.
    o = online.finish()
    output_transcript(o, now=now)
