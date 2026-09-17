#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
import warnings
from unittest.mock import AsyncMock, Mock

from pydantic import ValidationError

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InputDTMFFrame,
    InputTransportMessageFrame,
    InputTransportStartAudioStreamingFrame,
)
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor


class TestRTVIIncomingMessages(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.processor = RTVIProcessor()
        self.processor._message_queue = asyncio.Queue()
        self.processor.push_frame = AsyncMock()

    async def asyncTearDown(self):
        await self.processor.cleanup()

    async def test_peer_notifications_are_not_dispatched_as_client_requests(self):
        for message_type in (
            "error",
            "error-response",
            "server-message",
            "server-response",
            "metrics",
            "system-log",
            "bot-ready",
            "bot-started-speaking",
            "bot-stopped-speaking",
            "bot-output",
            "bot-llm-search-response",
            "user-transcription",
            "user-llm-text",
            "vad-user-started-speaking",
            "llm-function-call",
            "llm-function-call-started",
            "llm-function-call-in-progress",
            "llm-function-call-stopped",
            "ui-command",
            "ui-job-group",
        ):
            for has_id in (False, True):
                with self.subTest(type=message_type, has_id=has_id):
                    payload = {"label": "rtvi-ai", "type": message_type, "data": {}}
                    if has_id:
                        payload["id"] = "another-bots-request"
                    await self.processor._handle_transport_message(
                        InputTransportMessageFrame(message=payload)
                    )
                    self.assertTrue(self.processor._message_queue.empty())
                    self.processor.push_frame.assert_not_awaited()

    async def test_client_requests_still_reach_dispatch(self):
        for message_type in (
            "client-ready",
            "client-message",
            "disconnect-bot",
            "send-text",
            "dtmf",
            "raw-audio",
            "raw-audio-batch",
            "llm-function-call-result",
            "ui-event",
            "ui-snapshot",
            "ui-cancel-job-group",
        ):
            with self.subTest(type=message_type):
                await self.processor._handle_transport_message(
                    InputTransportMessageFrame(
                        message={"label": "rtvi-ai", "type": message_type, "id": "request-1"}
                    )
                )
                request = self.processor._message_queue.get_nowait()
                self.assertEqual((request.type, request.id), (message_type, "request-1"))
                self.processor._message_queue.task_done()
                self.processor.push_frame.assert_not_awaited()

    async def test_missing_client_id_reports_once_without_starting_a_peer_error_loop(self):
        await self.processor._handle_transport_message(
            InputTransportMessageFrame(message={"label": "rtvi-ai", "type": "client-ready"})
        )
        self.assertTrue(self.processor._message_queue.empty())
        self.processor.push_frame.assert_awaited_once()
        error = self.processor.push_frame.call_args.args[0].message
        self.assertEqual(error["type"], "error")
        self.assertIn("Field required", error["data"]["error"])

        peer = RTVIProcessor()
        peer._message_queue = asyncio.Queue()
        peer.push_frame = AsyncMock()
        try:
            await peer._handle_transport_message(InputTransportMessageFrame(message=error))
            peer.push_frame.assert_not_awaited()
            self.assertTrue(peer._message_queue.empty())
        finally:
            await peer.cleanup()


class TestRTVIClientReadyVersionHandling(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.processor = RTVIProcessor()

    async def asyncTearDown(self):
        await self.processor.cleanup()

    async def _call_handle_client_ready(self, data):
        """Helper to call _handle_client_ready with a mocked _send_error_response."""
        self.processor._send_error_response = AsyncMock()
        self.processor.set_client_ready = AsyncMock()
        await self.processor._handle_client_ready("req-1", data)

    # -- Fully compatible versions (protocol major 2) -------------------------

    async def test_valid_version_2_0_0_sends_no_error(self):
        data = RTVI.ClientReadyData(
            version="2.0.0",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor._send_error_response.assert_not_called()
        self.assertEqual(self.processor._client_version, [2, 0, 0])

    async def test_valid_version_2_3_1_sends_no_error(self):
        data = RTVI.ClientReadyData(
            version="2.3.1",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor._send_error_response.assert_not_called()
        self.assertEqual(self.processor._client_version, [2, 3, 1])

    # -- Deprecated legacy version (1.4.x) ------------------------------------

    # TODO: enable this once RTVI 2.0.0 is supported by all our client SDKs, and we start to emit the warnings again.
    # async def test_legacy_version_1_4_0_sends_deprecation_warning(self):
    #     """1.4.x clients receive a deprecation warning but the connection is allowed."""
    #     data = RTVI.ClientReadyData(
    #         version="1.4.0",
    #         about=RTVI.AboutClientData(library="test-client"),
    #     )
    #     await self._call_handle_client_ready(data)
    #     self.processor._send_error_response.assert_called_once()
    #     warning_msg = self.processor._send_error_response.call_args[0][1]
    #     self.assertIn("deprecated", warning_msg)
    #     self.assertIn("1.4.0", warning_msg)
    #     self.assertEqual(self.processor._client_version, [1, 4, 0])

    async def test_legacy_version_sets_client_ready(self):
        """1.4.x clients still become client-ready despite the warning."""
        data = RTVI.ClientReadyData(
            version="1.4.0",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor.set_client_ready.assert_called_once()

    async def test_legacy_version_1_0_0_sets_client_ready(self):
        """Any 1.x client is accepted as legacy."""
        data = RTVI.ClientReadyData(
            version="1.0.0",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor._send_error_response.assert_not_called()
        self.processor.set_client_ready.assert_called_once()

    async def test_legacy_version_1_2_0_sets_client_ready(self):
        """Any 1.x client is accepted as legacy."""
        data = RTVI.ClientReadyData(
            version="1.2.0",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor._send_error_response.assert_not_called()
        self.processor.set_client_ready.assert_called_once()

    # -- Incompatible versions ------------------------------------------------

    async def test_version_below_1_0_0_sends_error(self):
        data = RTVI.ClientReadyData(
            version="0.3.0",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor._send_error_response.assert_called_once()
        error_msg = self.processor._send_error_response.call_args[0][1]
        self.assertIn("0.3.0", error_msg)
        self.assertIn("not compatible", error_msg)

    async def test_no_version_sends_error(self):
        """Client sends no data (data=None)."""
        await self._call_handle_client_ready(None)
        self.processor._send_error_response.assert_called_once()
        error_msg = self.processor._send_error_response.call_args[0][1]
        self.assertIn("unknown", error_msg)

    async def test_invalid_version_format_sends_error(self):
        bad_versions = ["not-a-version", "123", "1.2.3.0", "junk", "1.2"]
        for version in bad_versions:
            with self.subTest(version=version):
                data = RTVI.ClientReadyData(
                    version=version,
                    about=RTVI.AboutClientData(library="test-client"),
                )
                await self._call_handle_client_ready(data)
                self.processor._send_error_response.assert_called_once()
                error_msg = self.processor._send_error_response.call_args[0][1]
                self.assertIn("Invalid client version format", error_msg)
                self.assertIn(version, error_msg)

    async def test_error_message_includes_compatibility_warning(self):
        """Incompatible version errors should append the compatibility warning."""
        for version in ["0.9.9", "3.0.0"]:
            with self.subTest(version=version):
                data = RTVI.ClientReadyData(
                    version=version,
                    about=RTVI.AboutClientData(library="test-client"),
                )
                await self._call_handle_client_ready(data)
                error_msg = self.processor._send_error_response.call_args[0][1]
                self.assertIn("Compatibility issues may occur", error_msg)

    async def test_client_ready_is_set_even_on_version_error(self):
        """Client-ready state should be set regardless of version errors."""
        data = RTVI.ClientReadyData(
            version="0.3.0",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor.set_client_ready.assert_called_once()

    async def test_client_ready_is_set_when_no_data(self):
        await self._call_handle_client_ready(None)
        self.processor.set_client_ready.assert_called_once()

    async def test_client_ready_pushes_start_audio_streaming_frame(self):
        self.processor.push_frame = AsyncMock()
        await self._call_handle_client_ready(None)
        pushed = [c.args[0] for c in self.processor.push_frame.call_args_list]
        self.assertTrue(any(isinstance(f, InputTransportStartAudioStreamingFrame) for f in pushed))


class TestRTVIFrameBasedAudio(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        await self.processor.cleanup()

    async def test_transport_param_is_deprecated(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.processor = RTVIProcessor(transport=Mock())
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    async def test_audio_buffer_pushes_input_audio_frame_downstream(self):
        self.processor = RTVIProcessor()
        self.processor.push_frame = AsyncMock()
        await self.processor._handle_audio_buffer(
            {"base64Audio": "AAAA", "sampleRate": 16000, "numChannels": 1}
        )
        pushed = [c.args[0] for c in self.processor.push_frame.call_args_list]
        self.assertEqual(len(pushed), 1)
        self.assertIsInstance(pushed[0], InputAudioRawFrame)
        self.assertEqual(pushed[0].sample_rate, 16000)


class TestRTVIDTMF(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        if hasattr(self, "processor"):
            await self.processor.cleanup()

    async def test_dtmf_pushes_input_dtmf_frame_downstream(self):
        self.processor = RTVIProcessor()
        self.processor.push_frame = AsyncMock()
        await self.processor._handle_dtmf(RTVI.DTMFInputData(buttons=[KeypadEntry.ONE]))
        pushed = [c.args[0] for c in self.processor.push_frame.call_args_list]
        self.assertEqual(len(pushed), 1)
        self.assertIsInstance(pushed[0], InputDTMFFrame)
        self.assertEqual(pushed[0].button, KeypadEntry.ONE)

    async def test_dtmf_sequence_pushes_one_frame_per_key_in_order(self):
        self.processor = RTVIProcessor()
        self.processor.push_frame = AsyncMock()
        data = RTVI.DTMFInputData.model_validate({"buttons": ["1", "2", "#"]})
        await self.processor._handle_dtmf(data)
        pushed = [c.args[0] for c in self.processor.push_frame.call_args_list]
        self.assertEqual(len(pushed), 3)
        for frame in pushed:
            self.assertIsInstance(frame, InputDTMFFrame)
        self.assertEqual(
            [f.button for f in pushed], [KeypadEntry.ONE, KeypadEntry.TWO, KeypadEntry.POUND]
        )

    def test_dtmf_input_data_rejects_invalid_button(self):
        with self.assertRaises(ValidationError):
            RTVI.DTMFInputData.model_validate({"buttons": ["1", "Z"]})

    def test_dtmf_input_data_rejects_empty_list(self):
        with self.assertRaises(ValidationError):
            RTVI.DTMFInputData.model_validate({"buttons": []})

    def test_dtmf_input_data_rejects_legacy_button_field(self):
        with self.assertRaises(ValidationError):
            RTVI.DTMFInputData.model_validate({"button": "1"})


if __name__ == "__main__":
    unittest.main()
