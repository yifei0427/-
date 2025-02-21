"""version 1 先处理后播放，不存在卡顿问题"""

import numpy as np
import soundfile as sf
import librosa
import pyaudio
from pytsmod import wsola
import time


def load_audio(file_path):
    """加载音频文件并转换为单声道"""
    signal, sr = sf.read(file_path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)  # 转换为单声道
    return signal, sr


def calculate_bpm(signal, sr):
    """计算音频的BPM"""
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    return tempo[0] if tempo[0] > 0 else None


def apply_tsm(chunk, sr, alpha):
    """应用时间缩放（Time Scale Modification）"""
    return wsola(chunk, alpha)


def simulate_heart_rate(duration, start_hr, end_hr):
    """模拟心率数据"""
    if start_hr == end_hr:
        return np.array([start_hr])
    else:
        return np.linspace(start_hr, end_hr, int(duration))


def stream_audio(processed_signal, sr, chunk_size=1024):
    """使用 PyAudio 实时流播放音频"""
    p = pyaudio.PyAudio()

    # 打开流
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sr,
                    output=True)

    # 将音频流分块播放
    for i in range(0, len(processed_signal), chunk_size):
        chunk = processed_signal[i:i + chunk_size]
        stream.write(chunk.astype(np.float32).tobytes())

    # 停止流并关闭
    stream.stop_stream()
    stream.close()
    p.terminate()


def apply_window_and_overlap(signal, chunk_size, overlap):
    """应用加窗和重叠分割"""
    hop_size = chunk_size - overlap
    windows = []
    for i in range(0, len(signal) - chunk_size, hop_size):
        chunk = signal[i:i + chunk_size]
        windowed_chunk = chunk * np.hanning(chunk_size)  # 加窗
        windows.append(windowed_chunk)
    return windows, hop_size


def overlap_add(chunks, original_hop_size, alphas):
    """重叠相加法拼接音频块，动态调整 hop_size"""
    output_length = 0
    for i, chunk in enumerate(chunks):
        if i == 0:
            output_length += len(chunk)  # 第一个块直接累加
        else:
            hop_size = int(original_hop_size * alphas[i])
            output_length += hop_size

    processed_signal = np.zeros(output_length)
    current_position = 0

    for i, chunk in enumerate(chunks):
        if i == 0:
            processed_signal[:len(chunk)] += chunk
            current_position += len(chunk)
        else:
            hop_size = int(original_hop_size * alphas[i])
            start = current_position - len(chunk) + hop_size  # 重叠部分
            end = start + len(chunk)
            processed_signal[start:end] += chunk
            current_position += hop_size

    return processed_signal


def main():
    audio_file_path = 'chopin.mp3'  # 输入音频文件路径
    signal, sr = load_audio(audio_file_path)
    bpm = calculate_bpm(signal, sr)
    print(f"Input Music BPM: {bpm}")

    # 设置块大小和重叠
    chunk_size = int(sr * 2)  # 每块2秒
    overlap = int(sr * 1)  # 重叠1秒
    windows, original_hop_size = apply_window_and_overlap(signal, chunk_size, overlap)

    # 模拟心率数据（假设心率在120到180之间变化）
    heart_rate_data = simulate_heart_rate(len(windows), 120, 180)

    # 处理每个音频块并动态调整播放速度
    processed_signal = []
    alphas = []  # 存储每个块的时间缩放因子
    for window, heart_rate in zip(windows, heart_rate_data):
        print(f"Processing with Heart Rate: {heart_rate} BPM")
        alpha = bpm / heart_rate if bpm else 1  # 计算时间缩放因子
        processed_chunk = apply_tsm(window, sr, alpha)  # 应用时间缩放
        processed_signal.append(processed_chunk)  # 将处理后的块添加到列表
        alphas.append(alpha)  # 存储时间缩放因子

    # 使用重叠相加法拼接所有音频块
    processed_signal = overlap_add(processed_signal, original_hop_size, alphas)

    # 实时播放处理后的音频流
    print("Starting real-time audio playback...")
    stream_audio(np.array(processed_signal), sr)


if __name__ == "__main__":
    main()

"""version 2 处理的同时播放，不使用overlap，有炸麦声，后半部分明显卡顿"""
# import numpy as np
# import soundfile as sf
# import librosa
# import pyaudio
# from pytsmod import wsola
# import time
# from multiprocessing import Process, Queue
#
#
# def load_audio(file_path):
#     """加载音频文件并转换为单声道"""
#     signal, sr = sf.read(file_path)
#     if signal.ndim > 1:
#         signal = signal.mean(axis=1)  # 转换为单声道
#     return signal, sr
#
#
# def calculate_bpm(signal, sr):
#     """计算音频的BPM"""
#     onset_env = librosa.onset.onset_strength(y=signal, sr=sr)
#     tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
#     return tempo[0] if tempo[0] > 0 else None
#
#
# def apply_tsm(chunk, sr, alpha):
#     """应用时间缩放（Time Scale Modification）"""
#     return wsola(chunk, alpha)
#
#
# def simulate_heart_rate(duration, start_hr, end_hr):
#     """模拟心率数据"""
#     if start_hr == end_hr:
#         return np.array([start_hr])
#     else:
#         return np.linspace(start_hr, end_hr, int(duration))
#
#
# def stream_audio(queue, sr, chunk_size=1024):
#     """使用 PyAudio 实时流播放音频"""
#     p = pyaudio.PyAudio()
#
#     # 打开流
#     stream = p.open(format=pyaudio.paFloat32,
#                     channels=1,
#                     rate=sr,
#                     output=True)
#
#     while True:
#         chunk = queue.get()
#         if chunk is None:  # 结束信号
#             break
#         stream.write(chunk.astype(np.float32).tobytes())
#
#     # 停止流并关闭
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#
#
# def process_audio(queue, signal, sr, bpm, heart_rate_data):
#     """处理音频并发送到播放队列"""
#     chunk_size = int(sr * 1)  # 每秒处理一个片段
#
#     # 处理第一个片段：保持原速
#     first_chunk = signal[:chunk_size]
#     queue.put(first_chunk)  # 发送第一个片段
#     signal = signal[chunk_size:]  # 更新剩余音频信号
#
#     # 从第二个片段开始，逐秒处理音频并根据心率变速
#     for heart_rate in heart_rate_data:
#         alpha = bpm / heart_rate if bpm else 1  # 计算时间缩放因子
#         chunk = signal[:chunk_size]  # 取当前片段
#         processed_chunk = apply_tsm(chunk, sr, alpha)  # 应用时间缩放
#         queue.put(processed_chunk)  # 将处理后的片段发送到播放队列
#         signal = signal[chunk_size:]  # 更新剩余音频信号
#
#         # 输出当前的音乐播放速率
#         current_bpm = bpm / alpha
#         print(f"Current Music Play Rate (BPM): {current_bpm:.2f}")
#
#         time.sleep(1)  # 每秒处理一个片段
#
#     queue.put(None)  # 发送结束信号
#
#
# def main():
#     audio_file_path = 'chopin.mp3'  # 输入音频文件路径
#     signal, sr = load_audio(audio_file_path)
#     bpm = calculate_bpm(signal, sr)
#     print(f"Input Music BPM: {bpm}")
#
#     # 模拟心率数据（假设心率在120到180之间变化）
#     heart_rate_data = simulate_heart_rate(len(signal) // sr, 120, 180)
#
#     # 创建队列用于进程间通信
#     queue = Queue()
#
#     # 创建并启动音频播放进程
#     stream_audio_process = Process(target=stream_audio, args=(queue, sr))
#     stream_audio_process.start()
#
#     # 创建并启动音频处理进程
#     process_audio_process = Process(target=process_audio, args=(queue, signal, sr, bpm, heart_rate_data))
#     process_audio_process.start()
#
#     # 等待两个进程完成
#     process_audio_process.join()
#     stream_audio_process.join()
#
#
# if __name__ == "__main__":
#     main()


"""version 3 处理的同时播放，使用overlap平滑过渡，没有炸麦和卡顿声，但是overlap的片段只是单纯的重复，音乐声音忽大忽小"""
# import numpy as np
# import soundfile as sf
# import librosa
# import pyaudio
# from pytsmod import wsola
# import time
# from multiprocessing import Process, Queue
#
#
# def load_audio(file_path):
#     """加载音频文件并转换为单声道"""
#     signal, sr = sf.read(file_path)
#     if signal.ndim > 1:
#         signal = signal.mean(axis=1)  # 转换为单声道
#     return signal, sr
#
#
# def calculate_bpm(signal, sr):
#     """计算音频的BPM"""
#     onset_env = librosa.onset.onset_strength(y=signal, sr=sr)
#     tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
#     return tempo[0] if tempo[0] > 0 else None
#
#
# def apply_tsm(chunk, sr, alpha):
#     """应用时间缩放（Time Scale Modification）"""
#     return wsola(chunk, alpha)
#
#
# def simulate_heart_rate(duration, start_hr, end_hr):
#     """模拟心率数据"""
#     if start_hr == end_hr:
#         return np.array([start_hr])
#     else:
#         return np.linspace(start_hr, end_hr, int(duration))
#
#
# def stream_audio(queue, sr, chunk_size=1024):
#     """使用 PyAudio 实时流播放音频"""
#     p = pyaudio.PyAudio()
#
#     # 打开流
#     stream = p.open(format=pyaudio.paFloat32,
#                     channels=1,
#                     rate=sr,
#                     output=True)
#
#     while True:
#         chunk = queue.get()
#         if chunk is None:  # 结束信号
#             break
#         stream.write(chunk.astype(np.float32).tobytes())
#
#     # 停止流并关闭
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#
#
# def apply_window_and_overlap(signal, chunk_size, overlap):
#     """应用加窗和重叠分割"""
#     hop_size = chunk_size - overlap
#     windows = []
#     for i in range(0, len(signal) - chunk_size, hop_size):
#         chunk = signal[i:i + chunk_size]
#         windowed_chunk = chunk * np.hanning(chunk_size)  # 加窗
#         windows.append(windowed_chunk)
#     return windows, hop_size
#
#
# def overlap_add(chunks, original_hop_size, alphas):
#     """重叠相加法拼接音频块，动态调整 hop_size"""
#     output_length = 0
#     for i, chunk in enumerate(chunks):
#         if i == 0:
#             output_length += len(chunk)  # 第一个块直接累加
#         else:
#             hop_size = int(original_hop_size * alphas[i])
#             output_length += hop_size
#
#     processed_signal = np.zeros(output_length)
#     current_position = 0
#
#     for i, chunk in enumerate(chunks):
#         if i == 0:
#             processed_signal[:len(chunk)] += chunk
#             current_position += len(chunk)
#         else:
#             hop_size = int(original_hop_size * alphas[i])
#             start = current_position - len(chunk) + hop_size  # 重叠部分
#             end = start + len(chunk)
#             processed_signal[start:end] += chunk
#             current_position += hop_size
#
#     return processed_signal
#
#
# def process_audio(queue, signal, sr, bpm, heart_rate_data):
#     """处理音频并发送到播放队列"""
#     chunk_size = int(sr * 2)  # 每块2秒
#     overlap = int(sr * 1)  # 重叠1秒
#     windows, original_hop_size = apply_window_and_overlap(signal, chunk_size, overlap)
#
#     processed_signal = []
#     alphas = []  # 存储每个块的时间缩放因子
#
#     for heart_rate in heart_rate_data:
#         alpha = bpm / heart_rate if bpm else 1  # 计算时间缩放因子
#         window = windows.pop(0)  # 获取当前窗口
#         processed_chunk = apply_tsm(window, sr, alpha)  # 应用时间缩放
#         processed_signal.append(processed_chunk)  # 将处理后的块添加到列表
#         alphas.append(alpha)  # 存储时间缩放因子
#
#         # 输出当前的音乐播放速率
#         current_bpm = bpm / alpha
#         print(f"Processing with Heart Rate: {heart_rate} BPM, Music Play Rate: {current_bpm:.2f}")
#
#         # 每秒处理一个片段并将其放入队列
#         if len(processed_signal) > 0:
#             queue.put(processed_signal[-1])  # 将最近的一个处理后的块放入队列
#
#         time.sleep(1)  # 每秒读取一次心率值
#
#     # 使用重叠相加法拼接所有音频块
#     processed_signal = overlap_add(processed_signal, original_hop_size, alphas)
#
#     # 将音频数据传输到播放队列
#     for chunk in processed_signal:
#         queue.put(chunk)
#     queue.put(None)  # 结束信号
#
#
# def main():
#     audio_file_path = 'chopin.mp3'  # 输入音频文件路径
#     signal, sr = load_audio(audio_file_path)
#     bpm = calculate_bpm(signal, sr)
#     print(f"Input Music BPM: {bpm}")
#
#     # 模拟心率数据（假设心率在120到180之间变化）
#     heart_rate_data = simulate_heart_rate(len(signal) // sr, 120, 180)
#
#     # 创建队列用于进程间通信
#     queue = Queue()
#
#     # 创建并启动音频播放进程
#     stream_audio_process = Process(target=stream_audio, args=(queue, sr))
#     stream_audio_process.start()
#
#     # 创建并启动音频处理进程
#     process_audio_process = Process(target=process_audio, args=(queue, signal, sr, bpm, heart_rate_data))
#     process_audio_process.start()
#
#     # 等待两个进程完成
#     process_audio_process.join()
#     stream_audio_process.join()
#
#
# if __name__ == "__main__":
#     main()