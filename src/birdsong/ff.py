"""Locate working ffmpeg/ffprobe binaries (static-ffmpeg bundle)."""

from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise

FFMPEG, FFPROBE = get_or_fetch_platform_executables_else_raise()
