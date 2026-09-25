import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,
    device_map="mps:0",
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs=dict(
        dtype=torch.bfloat16,
        device_map="mps:0",
    ),
)

results = model.transcribe(
    #audio="wav/epi.wav",
    audio="wav/speech-and-music.wav",
    language=None, # set "English" to force the language
    return_time_stamps=True,
)

print(results[0].language)
print(results[0].text)

for time_stamp in results[0].time_stamps:
    print(time_stamp.text, time_stamp.start_time, time_stamp.end_time)
