#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Dubit provider boundaries preserve turn identity across overlapping completions."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from websockets.protocol import State

from pipecat.frames.frames import ControlFrame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.meta.stt import MetaSTTService
from pipecat.services.stt_service import STTService


class _PlainSTTService(STTService):
    async def run_stt(self, audio: bytes):
        yield None


class MetaTurnBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def service(self, **kwargs):
        service = MetaSTTService(api_key="offline-test", **kwargs)
        service._session_id = "session-1"
        service.push_frame = AsyncMock()
        service.emit_stt_usage_metrics = AsyncMock()
        service._trace_transcription = AsyncMock()
        return service

    def frames(self, service):
        return [call.args[0] for call in service.push_frame.await_args_list]

    def test_only_endpointing_advertises_native_turn_boundaries(self):
        self.assertIs(getattr(_PlainSTTService(), "supports_turn_boundaries", None), False)
        for mode, expected in (
            ("ENDPOINTING", True),
            ("PUSH_TO_TALK", False),
            ("DIARIZATION", False),
        ):
            with self.subTest(mode=mode):
                service = self.service(settings=MetaSTTService.Settings(mode=mode))
                self.assertIs(getattr(service, "supports_turn_boundaries", None), expected)

    async def test_native_boundaries_and_transcripts_remain_ordered_across_overlapping_turns(self):
        for first, second in ((1, 2), ("1", "2")):
            with self.subTest(turn_ids=(first, second)):
                service = self.service()
                messages = [
                    {"type": "speechStart", "turnId": first},
                    {"type": "transcript", "transcript": "first", "final": False},
                    {"type": "speechEnd", "turnId": first},
                    {"type": "speechStart", "turnId": second},
                    {"type": "transcript", "transcript": "second", "final": False},
                    {"type": "speechComplete", "turnId": first, "transcript": "First."},
                    {"type": "transcript", "transcript": "second again", "final": True},
                    {"type": "speechEnd", "turnId": second},
                    {"type": "speechComplete", "turnId": second, "transcript": "Second again."},
                ]
                for message in messages:
                    await service._handle_message(message)

                frames = self.frames(service)
                self.assertEqual(
                    [type(frame).__name__ for frame in frames],
                    [
                        "STTSpeechStartFrame",
                        "InterimTranscriptionFrame",
                        "STTSpeechEndFrame",
                        "STTSpeechStartFrame",
                        "InterimTranscriptionFrame",
                        "TranscriptionFrame",
                        "InterimTranscriptionFrame",
                        "STTSpeechEndFrame",
                        "TranscriptionFrame",
                    ],
                )
                turns = [first, first, first, second, second, first, second, second, second]
                for frame, message, turn in zip(frames, messages, turns):
                    if isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
                        self.assertEqual(
                            frame.metadata,
                            {"stt_session_id": "session-1", "stt_turn_id": turn},
                        )
                        self.assertIs(frame.result, message)
                        self.assertEqual(frame.text, message["transcript"])
                        if isinstance(frame, TranscriptionFrame):
                            self.assertTrue(frame.finalized)
                    else:
                        self.assertIsInstance(frame, ControlFrame)
                        self.assertEqual((frame.session_id, frame.turn_id), ("session-1", turn))

    async def test_late_speech_end_does_not_relabel_new_partial(self):
        service = self.service()
        for message in (
            {"type": "speechStart", "turnId": 1},
            {"type": "speechStart", "turnId": 2},
            {"type": "speechEnd", "turnId": 1},
            {"type": "transcript", "transcript": "new speech", "final": False},
        ):
            await service._handle_message(message)
        self.assertEqual(
            self.frames(service)[-1].metadata,
            {"stt_session_id": "session-1", "stt_turn_id": 2},
        )

    async def test_empty_completion_closes_its_turn_without_speaking_wrappers(self):
        service = self.service(vad_enabled=True)
        message = {"type": "speechComplete", "turnId": 1, "transcript": ""}
        await service._handle_message(message)
        frames = self.frames(service)
        self.assertEqual([type(frame) for frame in frames], [TranscriptionFrame])
        self.assertEqual(frames[0].text, "")
        self.assertTrue(frames[0].finalized)
        self.assertIs(frames[0].result, message)
        self.assertEqual(frames[0].metadata, {"stt_session_id": "session-1", "stt_turn_id": 1})
        service._trace_transcription.assert_not_awaited()

    async def test_client_delimited_transcripts_keep_metadata_and_raw_results(self):
        service = self.service(settings=MetaSTTService.Settings(mode="PUSH_TO_TALK"))
        messages = [
            {"type": "transcript", "transcript": "partial", "final": False},
            {"type": "transcript", "transcript": "Final.", "final": True, "turnId": "client-1"},
        ]
        for message in messages:
            await service._handle_message(message)
        frames = self.frames(service)
        self.assertEqual(
            [type(frame) for frame in frames], [InterimTranscriptionFrame, TranscriptionFrame]
        )
        for frame, message, turn in zip(frames, messages, (None, "client-1")):
            self.assertEqual(frame.metadata, {"stt_session_id": "session-1", "stt_turn_id": turn})
            self.assertIs(frame.result, message)
        self.assertTrue(frames[-1].finalized)

    async def test_disconnect_clears_partial_identity(self):
        service = self.service()
        await service._handle_message({"type": "speechStart", "turnId": 1})
        await service._disconnect_websocket()
        await service._handle_message({"type": "transcript", "transcript": "unassigned"})
        self.assertEqual(
            self.frames(service)[-1].metadata,
            {"stt_session_id": None, "stt_turn_id": None},
        )

    async def test_new_session_clears_partial_identity_and_distinguishes_reused_turn_ids(self):
        service = self.service()
        await service._handle_message({"type": "speechStart", "turnId": 1})
        await service._handle_message({"type": "transcript", "transcript": "old session"})
        old_frame = self.frames(service)[-1]
        socket = SimpleNamespace(
            state=State.OPEN,
            send=AsyncMock(),
            recv=AsyncMock(return_value=json.dumps({"sessionId": "session-2"})),
        )
        service._sample_rate = 16000
        service._websocket_connect = AsyncMock(return_value=socket)
        service._start_audio_sender = Mock()
        await service._connect_websocket()
        await service._handle_message({"type": "transcript", "transcript": "before start"})
        self.assertEqual(
            self.frames(service)[-1].metadata,
            {"stt_session_id": "session-2", "stt_turn_id": None},
        )
        await service._handle_message({"type": "speechStart", "turnId": 1})
        await service._handle_message({"type": "transcript", "transcript": "new session"})
        self.assertEqual(old_frame.metadata, {"stt_session_id": "session-1", "stt_turn_id": 1})
        self.assertEqual(
            self.frames(service)[-1].metadata,
            {"stt_session_id": "session-2", "stt_turn_id": 1},
        )


if __name__ == "__main__":
    unittest.main()
