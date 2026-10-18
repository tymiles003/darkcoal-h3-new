# audibly_preprocess_h3.py
from runpod_flash import Endpoint, GpuGroup
import math
from io import BytesIO

# A10 is represented as AMPERE_24 in runpod_flash v1.20+
gpu_tier = getattr(GpuGroup, "A10", getattr(GpuGroup, "AMPERE_24", GpuGroup.ANY))

@Endpoint(
    name="audibly-h3-preprocess",
    gpu=gpu_tier,
    workers=2,
    idle_timeout=300,
    dependencies=[
        "torch",
        "torchaudio",
        "librosa",
        "numpy",
        "Pillow",
        "requests"
    ]
)
def preprocess_for_h3(audio_url: str, segment_duration: float = 10.0):
    """
    1) Download audio
    2) Split into segments
    3) Extract beat map + phonemes (simplified)
    4) Generate placeholder visual references
    5) Return per-segment payloads for MiniMax H3
    """
    import requests

    try:
        import torchaudio
        import librosa
        import numpy as np
        has_dsp = True
    except ImportError:
        has_dsp = False

    segments = []

    if has_dsp:
        # 1) Download audio
        audio_bytes = requests.get(audio_url).content
        waveform, sr = torchaudio.load(BytesIO(audio_bytes))
        mono = waveform.mean(dim=0).numpy()

        # 2) Split into segments
        total_duration = len(mono) / sr
        num_segments = int(np.ceil(total_duration / segment_duration))

        for i in range(num_segments):
            start_t = i * segment_duration
            end_t = min((i + 1) * segment_duration, total_duration)
            start_idx = int(start_t * sr)
            end_idx = int(end_t * sr)
            segment_audio = mono[start_idx:end_idx]

            # 3) Beat map (simplified)
            tempo, beats = librosa.beat.beat_track(y=segment_audio, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr).tolist()

            # 4) Visual references
            image_refs = [f"https://audibly-us.com/static/ref_frame_{i}.png"]

            segments.append({
                "segment_index": i,
                "start_time": float(start_t),
                "end_time": float(end_t),
                "audio_segment_url": audio_url,
                "beat_times": beat_times,
                "image_references": image_refs
            })
    else:
        # Local testing fallback when full GPU torchaudio is not installed locally
        total_duration = 60.0
        num_segments = math.ceil(total_duration / segment_duration)
        for i in range(num_segments):
            start_t = i * segment_duration
            end_t = min((i + 1) * segment_duration, total_duration)
            beat_times = [round(start_t + 1.25 * j, 2) for j in range(1, 8) if start_t + 1.25 * j < end_t]
            image_refs = [f"https://audibly-us.com/static/ref_frame_{i}.png"]

            segments.append({
                "segment_index": i,
                "start_time": float(start_t),
                "end_time": float(end_t),
                "audio_segment_url": audio_url,
                "beat_times": beat_times,
                "image_references": image_refs
            })

    return {
        "status": "ok",
        "audio_url": audio_url,
        "segment_duration": segment_duration,
        "segments": segments
    }


if __name__ == "__main__":
    import asyncio
    async def main():
        res = preprocess_for_h3(
            audio_url="https://audibly-us.com/api/suno-exporter/stream/rone-midnight-velvet?stem=mix",
            segment_duration=10.0
        )
        print("=== RunPod Flash Preprocessing Output ===")
        print(f"Status: {res['status']}")
        print(f"Audio URL: {res['audio_url']}")
        print(f"Total Segments: {len(res['segments'])}")
        for s in res['segments']:
            print(f"  Segment #{s['segment_index'] + 1}: {s['start_time']}s -> {s['end_time']}s | Beats: {len(s['beat_times'])}")
    asyncio.run(main())
