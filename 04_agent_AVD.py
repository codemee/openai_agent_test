import asyncio
import sys
import time
import numpy as np
import numpy.typing as npt
import sounddevice as sd

from agents import Agent, WebSearchTool
from agents.voice import (
    StreamedAudioInput,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
)

from getchar import getkeys
from tools import shell_helper

# 隱藏游標
def hide_cursor():
    print("\r\033[?25l", end="")

# 顯示游標
def show_cursor():
    print("\r\033[?25h", end="")

agent = Agent(
    name="Assistant",
    instructions="你是說台灣中文的助理",
    model="gpt-4o-mini",
    tools=[
        WebSearchTool(),
        shell_helper,
    ],
)

CHUNK_LENGTH_S = 0.05  # 100ms
SAMPLE_RATE = 24000
FORMAT = np.int16
CHANNELS = 1

# Create an audio player using `sounddevice`
should_send_audio: asyncio.Event = asyncio.Event()
audio_input = StreamedAudioInput()

async def start_voice_pipeline() -> None:
    player = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=FORMAT)
    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))

    try:
        player.start()
        result = await pipeline.run(audio_input)

        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                player.write(event.data)
                print(
                    f"Received audio: "
                    f"{len(event.data) if event.data is not None else '0'} bytes"
                )
            elif event.type == "voice_stream_event_lifecycle":
                print(f"Lifecycle event: {event.event}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    finally:
        player.close()

async def send_mic_audio() -> None:
    read_size = int(SAMPLE_RATE * 0.02)

    stream = sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype="int16",
    )
    stream.start()

    try:
        while True:
            # 先累積基本的音訊資料
            if stream.read_available < read_size:
                await asyncio.sleep(0)
                continue

            # 等待按下 K 鍵才開始傳送音訊資料
            await should_send_audio.wait()

            data, _ = stream.read(read_size)

            # 傳送音訊資料給伺服端，伺服端會自動判斷段落就回應
            await audio_input.add_audio(data)
            await asyncio.sleep(0)
    except KeyboardInterrupt:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop()
        stream.close()


async def main() -> None:
    mic_task = asyncio.create_task(send_mic_audio())
    realtime_task = asyncio.create_task(start_voice_pipeline())

    is_recording = False
    print("\r⏹", end="")
    hide_cursor()
    while True:
        keys = getkeys()
        if len(keys) == 0:            
            await asyncio.sleep(0.1)
            continue
        key = keys.pop().lower()
        if key == "k":
            is_recording = not is_recording
            if is_recording:
                print("\r⏺", end="")
                should_send_audio.set()
            else:
                should_send_audio.clear()
                print("\r⏹", end="")
        elif key == "q":
            break

    show_cursor()
    mic_task.cancel()
    realtime_task.cancel()
    await asyncio.gather(mic_task, realtime_task)

if __name__ == "__main__":
    asyncio.run(main())