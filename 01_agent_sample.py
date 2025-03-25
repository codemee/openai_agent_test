# 這是 OpenAI 原本自己的範例
# 只會執行一輪問答
import asyncio
import random
import time
import numpy as np
import numpy.typing as npt
import sounddevice as sd

from agents import (
    Agent,
    function_tool,
    WebSearchTool
)
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from getchar import getkeys

def record_audio() -> npt.NDArray[np.float32]:
    print("按任意鍵開始錄音，再按一次結束錄音")

    recording = False
    audio_buffer: list[np.NDArray[np.float32]] = []

    def _audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}\n")
        if recording:
            audio_buffer.append(indata.copy())

    # Open the audio stream with the callback.
    with sd.InputStream(samplerate=24000, 
                        channels=1, 
                        dtype=np.float32, 
                        callback=_audio_callback
    ):
        while True:
            keys = getkeys()
            if len(keys) > 0:
                recording = not recording
                if recording:
                    print("Recording started...\n")
                else:
                    print("Recording stopped.\n")
                    break
            time.sleep(0.01)

    # Combine recorded audio chunks.
    if audio_buffer:
        audio_data = np.concatenate(audio_buffer, axis=0)
    else:
        audio_data = np.empty((0,), dtype=np.float32)

    return audio_data

@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"[debug] get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."


spanish_agent = Agent(
    name="Enlish",
    handoff_description="A english speaking agent.",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. Speak in Enlish.",
    ),
    model="gpt-4o-mini",
)

agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        "你是說台灣中文的助理，如果使用者說的是西班牙語，就交棒給英語的 agent",
    ),
    model="gpt-4o-mini",
    handoffs=[spanish_agent],
    tools=[get_weather, WebSearchTool()],
)


async def main():
    pipeline = VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent)
    )
    # buffer = np.zeros(24000 * 3, dtype=np.int16)
    audio_input = AudioInput(buffer=record_audio())

    result = await pipeline.run(audio_input)

    # Create an audio player using `sounddevice`
    player = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
    player.start()

    # Play the audio stream as it comes in
    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            player.write(event.data)


if __name__ == "__main__":
    asyncio.run(main())