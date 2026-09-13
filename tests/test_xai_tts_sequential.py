#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sequential-utterance regression tests for xAI streaming TTS."""

import asyncio
import base64
import json
import unittest
from unittest.mock import AsyncMock

from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSTextFrame,
)
from pipecat.services.xai.tts import XAITTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.asyncio.task_manager import TaskManager


class _SequentialXAIWebSocket:
    """Provider double that rejects a new utterance before ``audio.done``."""

    def __init__(self, *, with_timestamps: bool = False):
        self.state = State.OPEN
        self._incoming = asyncio.Queue()
        self._text = ""
        self._generating = False
        self._generation_tasks = []
        self._with_timestamps = with_timestamps

    async def send(self, raw: str):
        message = json.loads(raw)
        message_type = message.get("type")
        if message_type == "text.delta":
            if self._generating:
                await self._incoming.put(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "new utterance received before audio.done",
                        }
                    )
                )
                return
            self._text += message["delta"]
        elif message_type == "text.done":
            if self._generating:
                await self._incoming.put(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "new utterance received before audio.done",
                        }
                    )
                )
                return
            text = self._text
            self._text = ""
            self._generating = True
            self._generation_tasks.append(asyncio.create_task(self._emit_audio(text)))

    async def _emit_audio(self, text: str):
        await asyncio.sleep(0.01)
        marker = b"A\x00" if "first" in text else b"B\x00"
        event = {
            "type": "audio.delta",
            "delta": base64.b64encode(marker * 16).decode("ascii"),
        }
        if self._with_timestamps:
            word = "first" if "first" in text else "second"
            event["audio_timestamps"] = {
                "graph_chars": list(word),
                "graph_times": [[i * 0.1, (i + 1) * 0.1] for i in range(len(word))],
            }
        await self._incoming.put(json.dumps(event))
        await self._incoming.put(json.dumps({"type": "audio.done", "trace_id": text}))
        self._generating = False

    async def close(self):
        self.state = State.CLOSED
        if self._generation_tasks:
            await asyncio.gather(*self._generation_tasks)
        await self._incoming.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self._incoming.get()
        if message is None:
            raise StopAsyncIteration
        return message


class _ControlledXAIWebSocket:
    """Minimal duplex socket for driving server events independently."""

    def __init__(self, *, on_send=None):
        self.state = State.OPEN
        self.incoming = asyncio.Queue()
        self.sent = asyncio.Queue()
        self.on_send = on_send

    async def send(self, raw: str):
        message = json.loads(raw)
        await self.sent.put(message)
        if self.on_send:
            await self.on_send(self, message)

    async def emit(self, message: dict):
        await self.incoming.put(json.dumps(message))

    async def close(self):
        self.state = State.CLOSED
        await self.incoming.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self.incoming.get()
        if message is None:
            raise StopAsyncIteration
        return message


class XAITTSSequentialTests(unittest.IsolatedAsyncioTestCase):
    async def _run_pipeline(
        self,
        websocket,
        frames,
        *,
        settings=None,
        reconnect_on_error=True,
        delay_audio_context=False,
        stop_frame_timeout_s=0.2,
    ):
        kwargs = {}
        if settings is not None:
            kwargs["settings"] = settings
        tts = XAITTSService(
            api_key="offline-test",
            sample_rate=24000,
            stop_frame_timeout_s=stop_frame_timeout_s,
            reconnect_on_error=reconnect_on_error,
            **kwargs,
        )
        if isinstance(websocket, list):
            tts._websocket_connect = AsyncMock(side_effect=websocket)
        else:
            tts._websocket_connect = AsyncMock(return_value=websocket)
        if delay_audio_context:
            handle_audio_context = tts._handle_audio_context

            async def delayed_audio_context(context_id: str):
                await asyncio.sleep(0.05)
                await handle_audio_context(context_id)

            tts._handle_audio_context = delayed_audio_context

        down_frames, up_frames = await asyncio.wait_for(
            run_test(tts, frames_to_send=frames), timeout=2.0
        )
        return tts, down_frames, up_frames

    async def _start_wire_tasks(self):
        websocket = _ControlledXAIWebSocket()
        tts = XAITTSService(api_key="offline-test", sample_rate=24000)
        tts._task_manager = TaskManager()
        tts._websocket = websocket
        tts._send_task = tts.create_task(tts._send_task_handler())
        receive_task = tts.create_task(tts._receive_xai_messages())
        self.addAsyncCleanup(self._stop_wire_tasks, tts, websocket, receive_task)
        return tts, websocket

    async def _stop_wire_tasks(self, tts, websocket, receive_task):
        await websocket.close()
        await receive_task
        if tts._send_task:
            await tts.cancel_task(tts._send_task)
            tts._send_task = None

    async def _assert_no_sent_message(self, websocket):
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(websocket.sent.get(), timeout=0.02)

    async def _wait_until(self, predicate, timeout=0.3):
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("condition was not met")
            await asyncio.sleep(0)

    async def test_clear_without_active_provider_context_blocks_next_utterance(self):
        tts, websocket = await self._start_wire_tasks()

        await tts.on_audio_context_interrupted("not-yet-sent")
        self.assertEqual(await websocket.sent.get(), {"type": "text.clear"})
        await tts._send_queue.put(("text.delta", "next", "next-context"))

        await self._assert_no_sent_message(websocket)
        await websocket.emit({"type": "audio.clear"})
        self.assertEqual(
            await asyncio.wait_for(websocket.sent.get(), timeout=0.2),
            {"type": "text.delta", "delta": "next"},
        )

    async def test_clear_before_text_done_blocks_next_context(self):
        tts, websocket = await self._start_wire_tasks()
        await tts._send_queue.put(("text.delta", "first", "first-context"))
        self.assertEqual(await websocket.sent.get(), {"type": "text.delta", "delta": "first"})

        await tts.on_audio_context_interrupted("first-context")
        self.assertEqual(await websocket.sent.get(), {"type": "text.clear"})
        await tts._send_queue.put(("text.delta", "next", "next-context"))

        await self._assert_no_sent_message(websocket)
        await websocket.emit({"type": "audio.clear"})
        self.assertEqual(
            await asyncio.wait_for(websocket.sent.get(), timeout=0.2),
            {"type": "text.delta", "delta": "next"},
        )

    async def test_clear_discards_a_next_context_already_dequeued_by_sender(self):
        tts, websocket = await self._start_wire_tasks()
        await tts._send_queue.put(("text.delta", "first", "first-context"))
        self.assertEqual(await websocket.sent.get(), {"type": "text.delta", "delta": "first"})
        await tts._send_queue.put(("text.delta", "cancelled", "cancelled-context"))
        while not tts._send_queue.empty():
            await asyncio.sleep(0)

        await tts.on_audio_context_interrupted("first-context")
        self.assertEqual(await websocket.sent.get(), {"type": "text.clear"})
        await websocket.emit({"type": "audio.clear"})

        await self._assert_no_sent_message(websocket)
        await tts._send_queue.put(("text.delta", "next", "next-context"))
        self.assertEqual(
            await asyncio.wait_for(websocket.sent.get(), timeout=0.2),
            {"type": "text.delta", "delta": "next"},
        )

    async def test_multiple_interrupted_contexts_send_one_clear(self):
        tts, websocket = await self._start_wire_tasks()
        await tts.on_audio_context_interrupted("first-context")
        self.assertEqual(await websocket.sent.get(), {"type": "text.clear"})

        await tts.on_audio_context_interrupted("queued-context")

        await self._assert_no_sent_message(websocket)

    async def test_error_during_clear_discards_queued_context_before_next_response(self):
        tts, websocket = await self._start_wire_tasks()
        await tts.on_audio_context_interrupted("first-context")
        self.assertEqual(await websocket.sent.get(), {"type": "text.clear"})
        await tts._send_queue.put(("text.delta", "cancelled", "cancelled-context"))
        await tts._send_queue.put(("text.done", "", "cancelled-context"))
        await asyncio.sleep(0)

        await websocket.emit({"type": "error", "message": "clear failed"})

        await self._assert_no_sent_message(websocket)
        await tts._send_queue.put(("text.delta", "next", "next-context"))
        self.assertEqual(
            await asyncio.wait_for(websocket.sent.get(), timeout=0.2),
            {"type": "text.delta", "delta": "next"},
        )

    async def test_late_audio_done_does_not_release_next_context_before_audio_clear(self):
        tts, websocket = await self._start_wire_tasks()
        await tts._send_queue.put(("text.delta", "first", "first-context"))
        await tts._send_queue.put(("text.done", "", "first-context"))
        self.assertEqual(await websocket.sent.get(), {"type": "text.delta", "delta": "first"})
        self.assertEqual(await websocket.sent.get(), {"type": "text.done"})

        await tts.on_audio_context_interrupted("first-context")
        self.assertEqual(await websocket.sent.get(), {"type": "text.clear"})
        await tts._send_queue.put(("text.delta", "next", "next-context"))
        await websocket.emit({"type": "audio.done"})

        await self._assert_no_sent_message(websocket)
        await websocket.emit({"type": "audio.clear"})
        self.assertEqual(
            await asyncio.wait_for(websocket.sent.get(), timeout=0.2),
            {"type": "text.delta", "delta": "next"},
        )

    async def test_flush_without_context_id_closes_current_provider_utterance(self):
        tts, websocket = await self._start_wire_tasks()
        await tts._send_queue.put(("text.delta", "first", "first-context"))
        self.assertEqual(await websocket.sent.get(), {"type": "text.delta", "delta": "first"})

        await tts.flush_audio()

        self.assertEqual(
            await asyncio.wait_for(websocket.sent.get(), timeout=0.2), {"type": "text.done"}
        )

    async def test_back_to_back_responses_wait_for_audio_done_and_keep_contexts_separate(self):
        websocket = _SequentialXAIWebSocket()
        _, down_frames, up_frames = await self._run_pipeline(
            websocket,
            [
                LLMFullResponseStartFrame(),
                LLMTextFrame("first response without terminal punctuation"),
                LLMFullResponseEndFrame(),
                LLMFullResponseStartFrame(),
                LLMTextFrame("second response also has no terminal punctuation"),
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.3),
            ],
            settings=XAITTSService.Settings(with_timestamps=False),
            delay_audio_context=True,
        )

        audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
        self.assertEqual([f.audio[:2] for f in audio_frames], [b"A\x00", b"B\x00"])
        self.assertEqual(len({f.context_id for f in audio_frames}), 2)
        self.assertFalse(any(isinstance(frame, ErrorFrame) for frame in [*down_frames, *up_frames]))

    async def test_end_frame_drains_queued_speak_without_an_explicit_delay(self):
        websocket = _SequentialXAIWebSocket()
        _, down_frames, up_frames = await self._run_pipeline(
            websocket,
            [TTSSpeakFrame(text="first")],
            settings=XAITTSService.Settings(with_timestamps=False),
        )

        audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
        self.assertEqual([f.audio[:2] for f in audio_frames], [b"A\x00"])
        self.assertFalse(any(isinstance(frame, ErrorFrame) for frame in [*down_frames, *up_frames]))

    async def test_default_timestamps_flush_final_words_to_their_own_contexts(self):
        websocket = _SequentialXAIWebSocket(with_timestamps=True)
        _, down_frames, up_frames = await self._run_pipeline(
            websocket,
            [
                LLMFullResponseStartFrame(),
                LLMTextFrame("first"),
                LLMFullResponseEndFrame(),
                LLMFullResponseStartFrame(),
                LLMTextFrame("second"),
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.2),
            ],
            delay_audio_context=True,
        )

        words = [f for f in down_frames if isinstance(f, TTSTextFrame)]
        self.assertEqual([f.text.strip() for f in words], ["first", "second"])
        self.assertEqual(len({f.context_id for f in words}), 2)
        self.assertFalse(any(isinstance(frame, ErrorFrame) for frame in [*down_frames, *up_frames]))

    async def test_provider_error_clears_current_and_queued_contexts(self):
        async def emit_error(websocket, message):
            if message["type"] == "text.done":
                await websocket.emit({"type": "error", "message": "provider failure"})

        websocket = _ControlledXAIWebSocket(on_send=emit_error)
        tts, down_frames, up_frames = await self._run_pipeline(
            websocket,
            [
                LLMFullResponseStartFrame(),
                LLMTextFrame("first"),
                LLMFullResponseEndFrame(),
                LLMFullResponseStartFrame(),
                LLMTextFrame("second"),
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.1),
            ],
            settings=XAITTSService.Settings(with_timestamps=False),
        )

        errors = [f.error for f in [*down_frames, *up_frames] if isinstance(f, ErrorFrame)]
        self.assertTrue(any("provider failure" in error for error in errors))
        self.assertEqual(tts.get_audio_contexts(), [])
        self.assertIsNone(tts._send_task)
        self.assertIsNone(tts._receive_task)

    async def test_cancel_discards_queued_text_and_stops_socket_tasks(self):
        websocket = _ControlledXAIWebSocket()
        tts = XAITTSService(api_key="offline-test", sample_rate=24000)
        tts._task_manager = TaskManager()
        tts._websocket = websocket
        tts._send_task = tts.create_task(tts._send_task_handler())
        tts._receive_task = tts.create_task(tts._receive_xai_messages())
        await tts._send_queue.put(("text.delta", "first", "first-context"))
        await tts._send_queue.put(("text.done", "", "first-context"))
        self.assertEqual(await websocket.sent.get(), {"type": "text.delta", "delta": "first"})
        self.assertEqual(await websocket.sent.get(), {"type": "text.done"})
        await tts._send_queue.put(("text.delta", "second", "second-context"))

        await asyncio.wait_for(tts.cancel(CancelFrame()), timeout=0.2)

        self.assertTrue(tts._send_queue.empty())
        self.assertIsNone(tts._send_task)
        self.assertIsNone(tts._receive_task)

    async def test_missing_audio_done_replaces_socket_and_sends_next_response(self):
        async def emit_audio_without_done(websocket, message):
            if message["type"] == "text.done":
                await websocket.emit(
                    {
                        "type": "audio.delta",
                        "delta": base64.b64encode(b"A\x00" * 16).decode("ascii"),
                    }
                )

        stalled = _ControlledXAIWebSocket(on_send=emit_audio_without_done)
        recovered = _SequentialXAIWebSocket()
        tts, down_frames, up_frames = await self._run_pipeline(
            [stalled, recovered],
            [
                LLMFullResponseStartFrame(),
                LLMTextFrame("first"),
                LLMFullResponseEndFrame(),
                LLMFullResponseStartFrame(),
                LLMTextFrame("second"),
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.3),
            ],
            settings=XAITTSService.Settings(with_timestamps=False),
            stop_frame_timeout_s=0.05,
        )

        audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
        self.assertEqual([f.audio[:2] for f in audio_frames], [b"A\x00", b"B\x00"])
        errors = [f.error for f in [*down_frames, *up_frames] if isinstance(f, ErrorFrame)]
        self.assertTrue(any("audio.done" in error and "timed out" in error for error in errors))
        self.assertEqual(tts._websocket_connect.await_count, 2)

    async def test_missing_audio_clear_replaces_socket_before_next_response(self):
        stalled = _ControlledXAIWebSocket()

        async def finish_recovered_utterance(websocket, message):
            if message["type"] == "text.done":
                await websocket.emit({"type": "audio.done"})

        recovered = _ControlledXAIWebSocket(on_send=finish_recovered_utterance)
        tts = XAITTSService(api_key="offline-test", sample_rate=24000, stop_frame_timeout_s=0.05)
        tts._task_manager = TaskManager()
        tts._websocket_connect = AsyncMock(return_value=recovered)
        tts._websocket = stalled
        tts._send_task = tts.create_task(tts._send_task_handler())
        tts._receive_task = tts.create_task(tts._receive_messages())

        await tts.on_audio_context_interrupted("first-context")
        self.assertEqual(await stalled.sent.get(), {"type": "text.clear"})
        await tts._send_queue.put(("text.delta", "second", "second-context"))
        await tts._send_queue.put(("text.done", "", "second-context"))
        await self._wait_until(lambda: recovered.sent.qsize() == 2)

        self.assertEqual(await recovered.sent.get(), {"type": "text.delta", "delta": "second"})
        self.assertEqual(await recovered.sent.get(), {"type": "text.done"})
        self.assertIs(tts._websocket, recovered)
        await tts._disconnect()

    async def test_audio_progress_extends_done_inactivity_timeout(self):
        async def emit_progress(websocket, message):
            if message["type"] != "text.done":
                return

            async def stream_audio():
                for _ in range(4):
                    await asyncio.sleep(0.03)
                    await websocket.emit(
                        {
                            "type": "audio.delta",
                            "delta": base64.b64encode(b"P\x00" * 16).decode("ascii"),
                        }
                    )
                await websocket.emit({"type": "audio.done"})

            asyncio.create_task(stream_audio())

        websocket = _ControlledXAIWebSocket(on_send=emit_progress)
        tts, down_frames, up_frames = await self._run_pipeline(
            websocket,
            [TTSSpeakFrame(text="progressing")],
            settings=XAITTSService.Settings(with_timestamps=False),
            stop_frame_timeout_s=0.05,
        )

        audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
        self.assertEqual(len(audio_frames), 4)
        self.assertFalse(any(isinstance(frame, ErrorFrame) for frame in [*down_frames, *up_frames]))
        self.assertEqual(tts._websocket_connect.await_count, 1)

    async def test_provider_disconnect_releases_current_and_queued_contexts(self):
        async def close_on_done(websocket, message):
            if message["type"] == "text.done":
                websocket.state = State.CLOSED
                await websocket.incoming.put(None)

        websocket = _ControlledXAIWebSocket(on_send=close_on_done)
        tts, _, _ = await self._run_pipeline(
            websocket,
            [
                LLMFullResponseStartFrame(),
                LLMTextFrame("first"),
                LLMFullResponseEndFrame(),
                LLMFullResponseStartFrame(),
                LLMTextFrame("second"),
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.1),
            ],
            settings=XAITTSService.Settings(with_timestamps=False),
            reconnect_on_error=False,
        )

        self.assertEqual(tts.get_audio_contexts(), [])
        self.assertIsNone(tts._send_task)
        self.assertIsNone(tts._receive_task)


if __name__ == "__main__":
    unittest.main()
