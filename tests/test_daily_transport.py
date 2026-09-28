#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Daily transport."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import BotConnectedFrame, CancelFrame, EndFrame, STTMetadataFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.stt_latency import DEEPGRAM_TTFS_P99
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.utils.asyncio.task_manager import TaskManager


def _make_transport(**params_kwargs) -> DailyTransport:
    with (
        patch("pipecat.transports.daily.transport.Daily"),
        patch("pipecat.transports.daily.transport.CallClient"),
    ):
        return DailyTransport(
            "https://mock.daily.co/mock", None, "bot", params=DailyParams(**params_kwargs)
        )


def _make_client_session(
    *, status: int = 200, body: str = "", error: Exception | None = None
) -> MagicMock:
    response = MagicMock(status=status)
    response.text = AsyncMock(return_value=body)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    if error:
        session.post.side_effect = error
    else:
        session.post.return_value = response

    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=None)
    return context


def _make_dialin_transport() -> DailyTransport:
    return _make_transport(
        api_key="test-api-key",
        dialin_settings={"call_id": "test-call", "call_domain": "test-domain"},
    )


@pytest.mark.asyncio
async def test_on_joined_pushes_stt_metadata_when_transcription_starts():
    transport = _make_transport(transcription_enabled=True)
    transport.start_transcription = AsyncMock(return_value=None)
    transport._input = AsyncMock()

    await transport._on_joined({})

    transport._input.push_stt_metadata_frame.assert_awaited_once()
    # BotConnectedFrame is pushed before the STT metadata frame.
    call_names = [name for (name, _, _) in transport._input.mock_calls]
    assert call_names.index("push_frame") < call_names.index("push_stt_metadata_frame")
    (frame,) = transport._input.push_frame.await_args.args
    assert isinstance(frame, BotConnectedFrame)


@pytest.mark.asyncio
async def test_on_joined_skips_stt_metadata_when_transcription_fails():
    transport = _make_transport(transcription_enabled=True)
    transport.start_transcription = AsyncMock(return_value="some error")
    transport._on_error = AsyncMock()
    transport._input = AsyncMock()

    await transport._on_joined({})

    transport._on_error.assert_awaited_once()
    transport._input.push_stt_metadata_frame.assert_not_awaited()
    transport._input.push_frame.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_joined_skips_stt_metadata_when_transcription_disabled():
    transport = _make_transport()
    transport._input = AsyncMock()

    await transport._on_joined({})

    transport._input.push_stt_metadata_frame.assert_not_awaited()
    transport._input.push_frame.assert_awaited_once()


@pytest.mark.asyncio
async def test_push_stt_metadata_frame_contents():
    transport = _make_transport(transcription_enabled=True)
    input_transport = transport.input()
    input_transport.broadcast_frame = AsyncMock()

    await input_transport.push_stt_metadata_frame()

    input_transport.broadcast_frame.assert_awaited_once_with(
        STTMetadataFrame,
        service_name=input_transport.name,
        ttfs_p99_latency=DEEPGRAM_TTFS_P99,
    )


@pytest.mark.asyncio
async def test_on_dialin_ready_reports_http_error_and_ready_event():
    transport = _make_dialin_transport()
    transport._on_error = AsyncMock()
    transport._call_event_handler = AsyncMock()
    session = _make_client_session(status=503, body="service unavailable")

    with patch("pipecat.transports.daily.transport.aiohttp.ClientSession", return_value=session):
        await transport._on_dialin_ready("sip:test@example.com")

    transport._on_error.assert_awaited_once_with(
        "Unable to handle dialin-ready event (status: 503, error: service unavailable)"
    )
    transport._call_event_handler.assert_awaited_once_with(
        "on_dialin_ready", "sip:test@example.com"
    )


@pytest.mark.asyncio
async def test_on_dialin_ready_reports_timeout():
    transport = _make_dialin_transport()
    transport._on_error = AsyncMock()
    session = _make_client_session(error=TimeoutError())

    with patch("pipecat.transports.daily.transport.aiohttp.ClientSession", return_value=session):
        await transport._handle_dialin_ready("sip:test@example.com")

    transport._on_error.assert_awaited_once_with(
        "Timeout handling dialin-ready event (https://api.daily.co/v1/dialin/pinlessCallUpdate)"
    )


@pytest.mark.asyncio
async def test_on_dialin_ready_reports_unexpected_error():
    transport = _make_dialin_transport()
    transport._on_error = AsyncMock()
    session = _make_client_session(error=RuntimeError("connection failed"))

    with patch("pipecat.transports.daily.transport.aiohttp.ClientSession", return_value=session):
        await transport._handle_dialin_ready("sip:test@example.com")

    transport._on_error.assert_awaited_once_with(
        "Error handling dialin-ready event "
        "(https://api.daily.co/v1/dialin/pinlessCallUpdate): connection failed"
    )


@pytest.mark.asyncio
async def test_on_dialin_ready_does_not_report_success_as_error():
    transport = _make_dialin_transport()
    transport._on_error = AsyncMock()
    session = _make_client_session()

    with patch("pipecat.transports.daily.transport.aiohttp.ClientSession", return_value=session):
        await transport._handle_dialin_ready("sip:test@example.com")

    transport._on_error.assert_not_awaited()


class TestDailyOwnership(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.transport = _make_transport(transcription_enabled=False)
        self.client = self.transport._client
        self.native = self.client._client
        self.native.join.side_effect = lambda *args, completion, **kwargs: completion({}, None)
        self.native.leave.side_effect = lambda *, completion: completion(None)
        self.manager = TaskManager()
        self.setup = FrameProcessorSetup(
            clock=SystemClock(), task_manager=self.manager, pipeline_worker=None
        )
        self.input = self.transport.input()
        self.output = self.transport.output()

    async def asyncTearDown(self):
        for task in list(self.manager.current_tasks()):
            await self.manager.cancel_task(task)

    async def test_one_owner_cannot_release_another_owners_room(self):
        await asyncio.gather(self.input.setup(self.setup), self.output.setup(self.setup))
        await asyncio.gather(self.input.stop(EndFrame()), self.input.cancel(CancelFrame()))
        self.assertEqual(self.native.leave.call_count, 0)
        await self.output.stop(EndFrame())
        await asyncio.gather(self.input.cleanup(), self.output.cleanup())
        self.assertEqual(self.native.leave.call_count, 1)
        self.assertEqual(self.native.release.call_count, 1)

    async def test_cleanup_releases_room_without_terminal_frame(self):
        await asyncio.gather(self.input.setup(self.setup), self.output.setup(self.setup))
        await asyncio.gather(self.input.cleanup(), self.output.cleanup())
        self.assertEqual(self.native.leave.call_count, 1)
        self.assertEqual(self.native.release.call_count, 1)

    async def test_failed_leave_still_releases_client(self):
        await self.input.setup(self.setup)
        self.native.leave.side_effect = RuntimeError("native leave failed")
        with self.assertRaises(RuntimeError):
            await self.input.cleanup()
        self.assertEqual(self.native.release.call_count, 1)
        await self.input.cleanup()
        self.assertEqual(self.native.leave.call_count, 1)
        self.assertEqual(self.native.release.call_count, 1)

    async def test_interrupted_join_still_releases_client(self):
        entered = asyncio.Event()
        self.native.join.side_effect = lambda *args, **kwargs: entered.set()
        setups = [
            asyncio.create_task(owner.setup(self.setup)) for owner in (self.input, self.output)
        ]
        await entered.wait()
        await asyncio.sleep(0)
        for setup in setups:
            setup.cancel()
        await asyncio.gather(*setups, return_exceptions=True)
        await asyncio.gather(self.input.cleanup(), self.output.cleanup())
        self.assertEqual(self.native.release.call_count, 1)

    async def test_failed_native_setup_releases_partial_client(self):
        self.transport = _make_transport(
            audio_in_enabled=True, audio_in_user_tracks=False, transcription_enabled=False
        )
        self.client = self.transport._client
        self.native = self.client._client
        self.input = self.transport.input()
        with patch("pipecat.transports.daily.transport.Daily") as daily:
            daily.create_speaker_device.side_effect = RuntimeError("device failed")
            with self.assertRaises(RuntimeError):
                await self.input.setup(self.setup)
        await self.input.cleanup()
        self.assertEqual(self.native.release.call_count, 1)

    async def test_missing_leave_callback_times_out_cancel_and_releases_native_client(self):
        await self._run_missing_leave_callback(end_first=False)

    async def test_cancel_interrupts_end_with_missing_native_leave_callback(self):
        await self._run_missing_leave_callback(end_first=True)

    async def _run_missing_leave_callback(self, *, end_first):
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineWorker
        from pipecat.workers.runner import WorkerRunner

        started = asyncio.Event()
        leaving = asyncio.Event()
        self.native.leave.side_effect = lambda **_: leaving.set()
        worker = PipelineWorker(
            Pipeline([self.input, self.output]), enable_rtvi=False, cancel_timeout_secs=0.03
        )
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)

        @worker.event_handler("on_pipeline_started")
        async def on_started(*_):
            started.set()

        with patch("pipecat.pipeline.worker.warm_deferred_imports"):
            running = asyncio.create_task(runner.run())
            await asyncio.wait_for(started.wait(), 1)
            if end_first:
                await worker.queue_frame(EndFrame())
                await asyncio.wait_for(leaving.wait(), 1)
            await runner.cancel(reason="owner disconnected")
            await asyncio.wait_for(running, 1)
        self.assertTrue(worker.has_finished())
        self.assertEqual(self.native.leave.call_count, 1)
        self.assertEqual(self.native.release.call_count, 1)

    async def test_failed_native_release_is_not_retried_by_either_owner(self):
        await asyncio.gather(self.input.setup(self.setup), self.output.setup(self.setup))
        self.native.release.side_effect = RuntimeError("release failed")
        results = await asyncio.gather(
            self.input.cleanup(), self.output.cleanup(), return_exceptions=True
        )
        self.assertEqual(sum(isinstance(result, RuntimeError) for result in results), 1)
        await asyncio.gather(self.input.cleanup(), self.output.cleanup())
        self.assertEqual(self.native.leave.call_count, 1)
        self.assertEqual(self.native.release.call_count, 1)

    async def test_one_owner_cannot_release_another_owners_client(self):
        await asyncio.gather(self.input.setup(self.setup), self.output.setup(self.setup))
        await self.input.cleanup()
        await self.input.cleanup()
        self.assertEqual(self.native.release.call_count, 0)
        self.assertEqual(self.native.leave.call_count, 0)
        await self.output.cleanup()
        self.assertEqual(self.native.release.call_count, 1)
        self.assertEqual(self.native.leave.call_count, 1)

    async def test_cancel_before_run_releases_constructor_owned_native_client(self):
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineWorker, WorkerParams

        worker = PipelineWorker(Pipeline([self.input, self.output]), enable_rtvi=False)
        await worker.cancel(reason="owner left before setup")
        await worker.run(WorkerParams(task_manager=self.manager))
        self.assertTrue(worker.has_finished())
        self.assertEqual(self.native.join.call_count, 0)
        self.assertEqual(self.native.release.call_count, 1)
        await asyncio.gather(self.input.cleanup(), self.output.cleanup())
        self.assertEqual(self.native.release.call_count, 1)

    async def test_uninitialized_owner_cleanup_does_not_release_active_owner_client(self):
        await self.output.setup(self.setup)
        await self.input.cleanup()
        self.assertEqual(self.native.release.call_count, 0)
        await self.output.cleanup()
        self.assertEqual(self.native.release.call_count, 1)
