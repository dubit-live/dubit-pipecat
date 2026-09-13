#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Dubit final-transcript boundaries must not turn partials into user turns."""

import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.services.meta.stt import MetaSTTService


class MetaDubitCompatTests(unittest.IsolatedAsyncioTestCase):
    def service(self, **kwargs):
        service = MetaSTTService(api_key="offline-test", **kwargs)
        service.push_frame = AsyncMock()
        service.emit_stt_usage_metrics = AsyncMock()
        service._trace_transcription = AsyncMock()
        return service

    async def test_standard_mode_emits_the_original_final_without_markers(self):
        service = self.service()
        message = {"type": "speechComplete", "turnId": 1, "transcript": "Hello"}
        await service._handle_message(message)
        frames = [call.args[0] for call in service.push_frame.await_args_list]
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], TranscriptionFrame)
        self.assertTrue(frames[0].finalized)
        self.assertIs(frames[0].result, message)

    async def test_natural_mode_wraps_each_final_with_standard_speaking_frames(self):
        service = self.service(vad_enabled=True)
        for turn_id, text in enumerate(("पहला हिस्सा", "और अगला हिस्सा")):
            await service._handle_message(
                {"type": "speechComplete", "turnId": turn_id, "transcript": text}
            )
        frames = [call.args[0] for call in service.push_frame.await_args_list]
        self.assertEqual(
            [type(frame) for frame in frames],
            [UserStartedSpeakingFrame, TranscriptionFrame, UserStoppedSpeakingFrame] * 2,
        )
        self.assertEqual(frames[1].text, "पहला हिस्सा")
        self.assertEqual(frames[4].text, "और अगला हिस्सा")
        self.assertTrue(frames[1].finalized)

    async def test_endpointing_partials_never_create_turns(self):
        service = self.service(vad_enabled=True)
        await service._handle_message({"type": "speechStart", "turnId": 1})
        await service._handle_message({"type": "transcript", "transcript": "Hello", "final": True})
        await service._handle_message({"type": "speechEnd", "turnId": 1})
        frames = [call.args[0] for call in service.push_frame.await_args_list]
        self.assertEqual([type(frame) for frame in frames], [InterimTranscriptionFrame])
