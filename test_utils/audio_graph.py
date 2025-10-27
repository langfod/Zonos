
import datetime
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import time
from utilities.audio_utils import get_cache_key
from utilities.cache_utils import get_wavout_dir

def plot_audio(wav_np, sr, words=None, audio_path=None):

    formatted_now_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    label = f"{formatted_now_time}_{get_cache_key(audio_path)}"
    # Create time axis
    t = np.linspace(0, len(wav_np) / sr, num=len(wav_np))
    wav_np = wav_np.squeeze(0).cpu().numpy()
    # Plot waveform
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(t, wav_np)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(label)
    ax.set_xlim(0, t[-1])
    ax.grid(True)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    # Plot word timestamps if provided
    if words:
        for idx, word in enumerate(words):
            ax.text(word.start, max(wav_np) * 0.9, word.word, fontsize=7, color='red')
            ax.axvline(x=word.start, color='r', linestyle='--')
            ax.axvline(x=word.end, color='g', linestyle='--')
    fig.tight_layout()
    path = get_wavout_dir().joinpath(f"{label}.png")

    fig.savefig(path,dpi=300)
    logger.info(f"Saved audio plot to {path}")
    #plt.show()