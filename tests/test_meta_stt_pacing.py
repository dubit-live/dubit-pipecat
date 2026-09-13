#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Cadence and lifecycle tests for Meta Muse realtime audio ingress."""

import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock

from websockets.protocol import State

from pipecat.frames.frames import CancelFrame, EndFrame
from pipecat.services.meta.stt import MetaSTTService
from pipecat.utils.asyncio.task_manager import TaskManager


class _FakeWebsocket:
    def __init__(self, *, wait_for_ack: bool = False, first_audio_delay: float = 0.0):
        self.state = State.OPEN
        self.sent: list[tuple[float, str | bytes]] = []
        self.closed = False
        self.handshake_sent = asyncio.Event()
        self.release_ack = asyncio.Event()
        self.first_audio_delay = first_audio_delay
        if not wait_for_ack:
            self.release_ack.set()

    async def send(self, payload):
        if isinstance(payload, bytes) and self.first_audio_delay:
            delay = self.first_audio_delay
            self.first_audio_delay = 0.0
            await asyncio.sleep(delay)
        self.sent.append((time.monotonic(), payload))
        if isinstance(payload, str) and not self.handshake_sent.is_set():
            self.handshake_sent.set()

    async def recv(self):
        await self.release_ack.wait()
        return json.dumps({"sessionId": "stream-test"})

    async def close(self):
        self.closed = True
        self.state = State.CLOSED


class MetaSTTPacingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.services: list[MetaSTTService] = []

    async def asyncTearDown(self):
        for service in self.services:
            await service._stop_audio_sender(drain=False, discard=True)
            await service._disconnect()

    async def connect(self, sample_rate=16000, *, wait_for_ack=False, first_audio_delay=0.0):
        service = MetaSTTService(api_key="offline-test")
        service._sample_rate = sample_rate
        service._task_manager = TaskManager()
        service.emit_stt_usage_metrics = AsyncMock()
        websocket = _FakeWebsocket(
            wait_for_ack=wait_for_ack,
            first_audio_delay=first_audio_delay,
        )
        service._websocket_connect = AsyncMock(return_value=websocket)
        self.services.append(service)
        connect_task = asyncio.create_task(service._connect_websocket())
        await websocket.handshake_sent.wait()
        return service, websocket, connect_task

    @staticmethod
    async def queue_audio(service: MetaSTTService, audio: bytes):
        async for _ in service.run_stt(audio):
            pass

    @staticmethod
    def binary_messages(websocket: _FakeWebsocket):
        return [(at, payload) for at, payload in websocket.sent if isinstance(payload, bytes)]

    async def test_silence_starts_only_after_ack_and_keeps_realtime_cadence(self):
        service, websocket, connect_task = await self.connect(wait_for_ack=True)

        await asyncio.sleep(0.03)
        self.assertEqual(self.binary_messages(websocket), [])

        websocket.release_ack.set()
        await connect_task
        await asyncio.sleep(0.07)

        packets = self.binary_messages(websocket)
        self.assertGreaterEqual(len(packets), 3)
        self.assertTrue(all(payload == b"\x00" * 640 for _, payload in packets))
        gaps = [right[0] - left[0] for left, right in zip(packets, packets[1:])]
        self.assertTrue(all(gap >= 0.012 for gap in gaps), gaps)

    async def test_real_audio_is_ordered_and_burst_input_is_paced(self):
        service, websocket, connect_task = await self.connect()
        await connect_task
        await asyncio.sleep(0.005)
        baseline = len(self.binary_messages(websocket))
        first = b"\x01" * 640
        second = b"\x02" * 640

        await self.queue_audio(service, first + second)
        await asyncio.sleep(0.055)

        packets = self.binary_messages(websocket)[baseline:]
        self.assertGreaterEqual(len(packets), 2)
        self.assertEqual([payload for _, payload in packets[:2]], [first, second])
        self.assertGreaterEqual(packets[1][0] - packets[0][0], 0.012)

        await asyncio.sleep(0.025)
        resumed_silence = self.binary_messages(websocket)[baseline + 2 :]
        self.assertTrue(any(payload == b"\x00" * 640 for _, payload in resumed_silence))

    async def test_one_delayed_send_recovers_the_elapsed_audio_budget(self):
        service, websocket, connect_task = await self.connect(first_audio_delay=0.05)
        started = time.monotonic()
        await connect_task
        await asyncio.sleep(0.075)

        packets = self.binary_messages(websocket)
        self.assertGreaterEqual(len(packets), 4)
        for index, (sent_at, _) in enumerate(packets):
            audio_duration = (index + 1) * 0.02
            self.assertLessEqual(audio_duration, sent_at - started + 0.021)

    async def test_packet_budget_tracks_16_and_24_khz_mono_pcm(self):
        for sample_rate, packet_bytes in ((16000, 640), (24000, 960)):
            with self.subTest(sample_rate=sample_rate):
                service, websocket, connect_task = await self.connect(sample_rate)
                await connect_task
                await asyncio.sleep(0.025)
                packets = self.binary_messages(websocket)
                self.assertGreaterEqual(len(packets), 1)
                self.assertTrue(all(len(payload) == packet_bytes for _, payload in packets))
                await service._stop_audio_sender(drain=False, discard=True)
                await service._disconnect()
                self.services.remove(service)

    async def test_usage_adds_only_the_synthetic_padding_duration(self):
        service, websocket, connect_task = await self.connect()
        service._stt_usage_pending_seconds = 1.25
        await connect_task
        await asyncio.sleep(0.045)

        padding = sum(len(payload) for _, payload in self.binary_messages(websocket))
        expected = 1.25 + padding / (16000 * 2)
        self.assertAlmostEqual(service._stt_usage_pending_seconds, expected)

    async def test_reconnect_stops_the_old_sender_and_restarts_after_new_ack(self):
        service, first_websocket, connect_task = await self.connect()
        await connect_task
        await asyncio.sleep(0.03)
        await service._disconnect_websocket()
        old_count = len(first_websocket.sent)

        second_websocket = _FakeWebsocket(wait_for_ack=True)
        service._websocket_connect = AsyncMock(return_value=second_websocket)
        reconnect_task = asyncio.create_task(service._connect_websocket())
        await second_websocket.handshake_sent.wait()
        await asyncio.sleep(0.025)
        self.assertEqual(self.binary_messages(second_websocket), [])

        second_websocket.release_ack.set()
        await reconnect_task
        await asyncio.sleep(0.03)
        self.assertGreaterEqual(len(self.binary_messages(second_websocket)), 1)
        self.assertEqual(len(first_websocket.sent), old_count)

    async def test_graceful_end_drains_real_audio_before_end_stream(self):
        service, websocket, connect_task = await self.connect()
        await connect_task
        await asyncio.sleep(0.005)
        baseline = len(self.binary_messages(websocket))
        audio = b"\x03" * (640 * 3)
        await self.queue_audio(service, audio)

        await service.stop(EndFrame())

        drained = self.binary_messages(websocket)[baseline:]
        self.assertEqual(b"".join(payload for _, payload in drained), audio)
        self.assertEqual(json.loads(websocket.sent[-1][1]), {"type": "endStream"})

    async def test_graceful_end_survives_sender_cancel_during_websocket_close(self):
        service, _, connect_task = await self.connect()
        await connect_task
        await self.queue_audio(service, b"\x05" * (640 * 10))
        stop_task = asyncio.create_task(service.stop(EndFrame()))
        await asyncio.sleep(0.03)

        await service._disconnect_websocket()
        await asyncio.wait_for(stop_task, timeout=0.2)

        self.assertTrue(service._disconnecting)
        buffered = len(service._audio_buffer)
        await self.queue_audio(service, b"late audio")
        self.assertEqual(len(service._audio_buffer), buffered)

    async def test_cancel_discards_buffer_and_never_writes_after_end_stream(self):
        service, websocket, connect_task = await self.connect()
        await connect_task
        await asyncio.sleep(0.005)
        await self.queue_audio(service, b"\x04" * (640 * 20))

        await service.cancel(CancelFrame())
        sent_at_cancel = list(websocket.sent)
        await asyncio.sleep(0.04)

        self.assertEqual(websocket.sent, sent_at_cancel)
        self.assertEqual(json.loads(websocket.sent[-1][1]), {"type": "endStream"})


if __name__ == "__main__":
    unittest.main()
