"""
语音转录工具
============
封装 speech_pipeline1 的说话人分离 + ASR 功能，供 Aegis 调用。

speech_pipeline1 路径通过环境变量 SPEECH_PIPELINE_PATH 配置，
默认为 E:/speech_pipeline1。模型首次加载约 30-60 秒，后续调用复用。
"""
import logging
import sys
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ===== torchaudio sox_effects 兼容性补丁 =====
# torchaudio >= 2.2 移除了 sox_effects 模块，但 ModelScope 的
# segmentation_clustering_pipeline 仍然依赖它做音频重采样。
# 此补丁用 torchaudio.functional.resample() 代替。
import torchaudio  # noqa: E402


def _patched_apply_effects_tensor(tensor, sample_rate, effects, channels_first=True):
    """用 torchaudio.functional.resample 替代 sox_effects 实现"""
    import torch
    if not effects:
        return tensor, sample_rate

    result = tensor
    result_sr = sample_rate
    # 确保是 2D: (channels, samples)
    if result.dim() == 1:
        result = result.unsqueeze(0)

    for effect in effects:
        if isinstance(effect, (list, tuple)) and len(effect) >= 2:
            if effect[0] == "rate":
                target_rate = int(effect[1])
                if result_sr != target_rate:
                    result = torchaudio.functional.resample(result, result_sr, target_rate)
                    result_sr = target_rate
            elif effect[0] == "channels":
                target_channels = int(effect[1])
                if result.shape[0] > target_channels:
                    result = result[:target_channels, :]
                elif result.shape[0] < target_channels:
                    result = result.repeat(target_channels, 1)

    return result, result_sr


# 修复合成的 sox_effects 模块（torchaudio.sox_effects 元素可不存在）
class _SoxEffectsModule:
    apply_effects_tensor = staticmethod(_patched_apply_effects_tensor)


if not hasattr(torchaudio, "sox_effects"):
    torchaudio.sox_effects = _SoxEffectsModule()
    logger.info("已激活 torchaudio sox_effects 兼容性补丁 (torchaudio %s)", torchaudio.__version__)
# ===== 补丁结束 =====

# 确保 speech_pipeline1 在 Python 路径中
# 通过环境变量 SPEECH_PIPELINE_PATH 配置，不再硬编码绝对路径
_PIPELINE_PATH = os.environ.get("SPEECH_PIPELINE_PATH", "")
if _PIPELINE_PATH and _PIPELINE_PATH not in sys.path:
    sys.path.insert(0, _PIPELINE_PATH)

_pipeline = None  # 懒加载单例


def _get_pipeline():
    """获取或初始化语音处理流水线（懒加载）"""
    global _pipeline
    if _pipeline is None:
        from pipeline import SpeechProcessingPipeline  # type: ignore
        config_path = os.environ.get("SPEECH_PIPELINE_CONFIG", os.path.join(_PIPELINE_PATH, "config.yaml"))
        logger.info("首次加载语音处理模型（约 30-60 秒）...")
        _pipeline = SpeechProcessingPipeline(config_path=config_path)
        logger.info("语音处理模型加载完成")
    return _pipeline


def _convert_to_wav(audio_path: str) -> str:
    """
    用 ffmpeg 将音频转为 16kHz 单声道 WAV，确保 pipeline 可靠读取。

    librosa 的 audioread 回退链在 uvicorn 线程环境中偶发失败
    （soundfile 不支持 MP3 → audioread/ffmpeg pipe 不稳定 → BytesIO 错误），
    提前用 ffmpeg 转换成标准 WAV 可彻底绕过此问题。
    """
    import subprocess
    import tempfile

    ext = os.path.splitext(audio_path)[1].lower()
    # WAV/FLAC 可直接被 soundfile 读取，无需转换
    if ext in (".wav", ".flac"):
        return audio_path

    wav_path = os.path.join(tempfile.gettempdir(), f"aegis_audio_{os.path.basename(audio_path)}.wav")
    logger.info(f"转换音频格式: {ext} → .wav ({wav_path})")

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", wav_path],
        capture_output=True, timeout=120,
    )
    if result.returncode != 0:
        error_msg = result.stderr.decode("utf-8", errors="ignore")[-300:]
        raise RuntimeError(f"ffmpeg 转换失败: {error_msg}")

    return wav_path


def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    """
    转录音频文件：说话人分离 + ASR + 标点恢复

    Args:
        audio_path: 音频文件路径

    Returns:
        {
            "audio_path": str,
            "audio_duration": float,
            "speakers": [str, ...],
            "full_text": str,
            "segments": [
                {"speaker": str, "start": float, "end": float, "text": str},
                ...
            ]
        }
    """
    pipeline = _get_pipeline()
    # 预转换为 WAV，绕过 librosa audioread 回退链的不稳定性
    wav_path = _convert_to_wav(audio_path)
    return pipeline.process_audio(wav_path)


def get_audio_model_status() -> Dict[str, Any]:
    """查询语音模型加载状态"""
    if _pipeline is None:
        return {"loaded": False, "diarizer": False, "recognizer": False}
    return {
        "loaded": True,
        "diarizer": _pipeline._diarizer is not None,
        "recognizer": _pipeline._recognizer is not None,
    }


def format_transcript_for_display(result: Dict[str, Any]) -> str:
    """将转录结果格式化为可读的 Markdown 文本"""
    lines = [
        f"## 语音转录结果\n",
        f"**文件**: {os.path.basename(result.get('audio_path', ''))}",
        f"**时长**: {result.get('audio_duration', 0):.1f} 秒",
        f"**说话人**: {', '.join(result.get('speakers', []))}\n",
        f"### 完整文本\n",
        result.get("full_text", ""),
        f"\n### 逐段详情\n",
    ]

    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "?")
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "")
        if text.strip():
            lines.append(f"- **{speaker}** [{start:.1f}s-{end:.1f}s]: {text}")

    return "\n".join(lines)
