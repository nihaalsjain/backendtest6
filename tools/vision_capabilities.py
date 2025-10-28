"""
VisionCapabilities mixin.

This mixin can be added to an agent class to provide vision support:
 - Receives images from the frontend over a byte-stream channel ("images")
 - Subscribes to remote video tracks (if present) and buffers frames
 - Attaches the most recent image/frame to the next user turn
 - Cleans up resources when the agent exits

Usage:
    class MyAgent(VisionCapabilities, Agent):
        def __init__(self):
            Agent.__init__(self, instructions="...")
            VisionCapabilities.__init__(self)

The lifecycle hooks (`on_enter`, `on_exit`, `on_user_turn_completed`) will
be called automatically by LiveKit during session operation.
"""

import asyncio
import base64
import time
from typing import Optional

from livekit import rtc
from livekit.agents import get_job_context
from livekit.agents.llm import ImageContent
from livekit.agents.utils.images import encode, EncodeOptions, ResizeOptions


class VisionCapabilities:
    """
    Drop-in vision helper that enables image/video input for an agent.

    Features:
      • Accepts images via byte-stream channel named "images"
      • Buffers the latest frame from a remote video track (with FPS throttling)
      • Injects the buffered frame into the next user turn
      • Cleans up streams and tasks when exiting

    Notes:
      - Call VisionCapabilities.__init__(self) inside your agent __init__.
      - This mixin assumes the agent has `self.chat_ctx` and `update_chat_ctx`
        available (inherited from LiveKit Agent).
    """

    def __init__(self) -> None:
        # Vision state
        self._latest_frame: Optional[str] = None  # Base64 data URL for LLM vision input
        self._video_stream: Optional[rtc.VideoStream] = None
        self._tasks: list[asyncio.Task] = []      # background tasks for handling streams
        self._last_frame_ts: float = 0.0
        self._frame_interval_s: float = 0.2       # throttle video → ~5 FPS
        self._max_upload_bytes: int = 8 * 1024 * 1024  # 8 MB upload cap

    # -------- lifecycle hooks --------
    async def on_enter(self):
        """
        Called when the agent enters a session.

        - Registers a byte-stream handler for "images" channel
        - Attaches to any existing remote video track
        - Subscribes to new video tracks dynamically
        """
        room = get_job_context().room

        # Byte-stream handler for frontend uploads
        def _image_received_handler(reader, participant_identity):
            task = asyncio.create_task(self._image_received(reader, participant_identity))
            self._tasks.append(task)
            task.add_done_callback(lambda t: self._safe_remove_task(t))

        room.register_byte_stream_handler("images", _image_received_handler)

        # Attach immediately if any participant already has a video track
        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                    self._create_video_stream(publication.track)
                    break

        # Subscribe to future video tracks
        @room.on("track_subscribed")
        def _on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self._create_video_stream(track)

    async def on_exit(self):
        """
        Called when the agent session exits.

        - Closes video stream if active
        - Cancels and cleans up all background tasks
        """
        if self._video_stream is not None:
            try:
                self._video_stream.close()
            except Exception:
                pass
            self._video_stream = None

        # Cancel and await all tasks
        for t in list(self._tasks):
            if not t.done():
                t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def on_user_turn_completed(self, turn_ctx, new_message: dict) -> None:
        """
        After each user turn completes, attach the latest video frame (if any)
        to the new message so the LLM sees the visual context.

        Resets `_latest_frame` after attaching so it won’t repeat.
        """
        if self._latest_frame:
            if isinstance(new_message.content, list):
                new_message.content.append(ImageContent(image=self._latest_frame))
            elif new_message.content is None:
                new_message.content = [ImageContent(image=self._latest_frame)]
            else:
                new_message.content = [new_message.content, ImageContent(image=self._latest_frame)]
            self._latest_frame = None

    # -------- internals --------
    async def _image_received(self, reader, participant_identity):
        """
        Handle images uploaded from the frontend via the "images" byte stream.

        - Buffers incoming chunks into memory (up to _max_upload_bytes)
        - Encodes the result as base64 data URL
        - Adds the image as a user message in the chat context
        """
        buf = bytearray()
        read_bytes = 0
        try:
            async for chunk in reader:
                buf.extend(chunk)
                read_bytes += len(chunk)
                if read_bytes > self._max_upload_bytes:
                    # Drop oversized upload silently (or log in future)
                    return
        except Exception:
            return

        # Default MIME type: PNG (frontend can add support for others)
        mime = "image/png"
        b64 = base64.b64encode(buf).decode("utf-8")

        # Add as a user message in the chat context
        chat_ctx = self.chat_ctx.copy()
        chat_ctx.add_message(
            role="user",
            content=[
                "Here's an image I want to share with you:",
                ImageContent(image=f"data:{mime};base64,{b64}"),
            ],
        )
        await self.update_chat_ctx(chat_ctx)

    def _create_video_stream(self, track: rtc.Track):
        """
        Subscribe to a remote video track and buffer the latest frame.

        - Uses an internal background task to read frames
        - Throttles to ~5 FPS
        - Encodes frames as JPEG base64 strings
        """
        # If already subscribed, close old stream
        if self._video_stream is not None:
            try:
                self._video_stream.close()
            except Exception:
                pass
            self._video_stream = None

        self._video_stream = rtc.VideoStream(track)

        async def _read_stream():
            try:
                async for event in self._video_stream:
                    now = time.monotonic()
                    # Throttle to frame_interval
                    if (now - self._last_frame_ts) < self._frame_interval_s:
                        continue
                    self._last_frame_ts = now

                    # Encode frame → JPEG → base64 string
                    image_bytes = encode(
                        event.frame,
                        EncodeOptions(
                            format="JPEG",
                            resize_options=ResizeOptions(
                                width=1024,
                                height=1024,
                                strategy="scale_aspect_fit",
                            ),
                        ),
                    )
                    self._latest_frame = (
                        "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")
                    )
            except asyncio.CancelledError:
                # Task was cancelled during shutdown
                pass
            except Exception:
                # Swallow errors to avoid killing the agent
                pass

        # Spawn reader task
        task = asyncio.create_task(_read_stream())
        self._tasks.append(task)
        task.add_done_callback(lambda t: self._safe_remove_task(t))

    def _safe_remove_task(self, t: asyncio.Task):
        """
        Safely remove a completed task from self._tasks.
        Ignores if the task was already removed.
        """
        try:
            self._tasks.remove(t)
        except ValueError:
            pass
