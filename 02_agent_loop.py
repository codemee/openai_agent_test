# 修改 OpenAI 的範例，可以進行多輪對答

import asyncio
import time
import numpy as np
import numpy.typing as npt
import sounddevice as sd

from agents import Agent, function_tool, WebSearchTool
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
)

from getchar import getkeys

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
    tools=[WebSearchTool()],
)


async def main():
    hide_cursor()
    print('按下 "r" 開始/結束錄音，按下 "q" 結束程式')
    print("\r⏹", end="")
    recording = False  # 是否錄音中
    audio_buffer: list[np.NDArray[np.float32]] = []  # 音訊暫存區

    # Create an audio player using `sounddevice`
    player = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
    player.start()

    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))

    def _audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}\n")
        if recording:
            audio_buffer.append(indata.copy())

    # Open the audio stream with the callback.
    with sd.InputStream(
        samplerate=24000, channels=1, dtype=np.float32, callback=_audio_callback
    ):
        while True:
            keys = getkeys()
            if len(keys) < 1:  # 沒有按鍵被按下
                time.sleep(0.01)
                continue
            key = keys[0].lower()
            if key == "q":
                break
            if key == "r":
                recording = not recording
                if recording:
                    print("\r⏺", end="")
                    continue
                print("\r⏹", end="")
                # 合併音訊片段
                if audio_buffer:
                    audio_data = np.concatenate(audio_buffer, axis=0)
                else:
                    audio_data = np.empty((0,), dtype=np.float32)
                audio_input = AudioInput(buffer=audio_data)

                result = await pipeline.run(audio_input)

                # 播放串流音訊
                status_ch = '▶'
                async for event in result.stream():
                    # print(event.type)
                    if event.type == "voice_stream_event_audio":
                        player.write(event.data)
                        print(f'\r{status_ch}', end="")
                        status_ch = '▶' if status_ch == ' ' else ' '
                print("\r⏹", end="")
                audio_buffer.clear()
    show_cursor()
    print("\r結束")
    player.stop()

if __name__ == "__main__":
    asyncio.run(main())
