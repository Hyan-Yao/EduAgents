import os
import re
import glob
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional


class VideoUtils:
    @staticmethod
    def extract_slides(pdf_path: str, output_dir: str, skip_first: bool = True) -> List[str]:
        """Extract PDF pages as PNG images at 200 DPI using PyMuPDF.

        skip_first=True drops the title page (page 0) so slide indices
        align with the script.md section numbering.
        """
        import fitz  # PyMuPDF

        os.makedirs(output_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        paths = []
        for page_num in range(len(doc)):
            if skip_first and page_num == 0:
                continue
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            img_path = os.path.join(output_dir, f"slide_{page_num:03d}.png")
            pix.save(img_path)
            paths.append(img_path)
        doc.close()
        return sorted(paths)

    @staticmethod
    def parse_script(script_md_path: str) -> List[str]:
        """Parse script.md into a list of per-slide narration strings.

        The ADDIE pipeline writes script.md in this format:
            # Slides Script: <name>
            ## Section N: <title>
            *(optional frame count)*
            <narration text>
            ---
        Returns one clean narration string per section, in order.
        """
        with open(script_md_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split on section headers; first element is the file-level header
        parts = re.split(r"^## Section \d+:.*$", text, flags=re.MULTILINE)
        narrations = []
        for part in parts[1:]:  # skip preamble before first section
            # Strip frame-count annotation and horizontal rules
            cleaned = re.sub(r"^\*\(\d+ frames?\)\*\s*", "", part.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"^---\s*$", "", cleaned, flags=re.MULTILINE)
            # Strip markdown headings, bold/italic markers
            cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", cleaned)
            cleaned = cleaned.strip()
            if cleaned:
                narrations.append(cleaned)
        return narrations

    @staticmethod
    def make_audio(
        text: str,
        output_path: str,
        use_openai: bool = True,
        api_key: Optional[str] = None,
        voice: str = "alloy",
    ) -> None:
        """Generate TTS audio. Uses OpenAI tts-1 by default; falls back to gTTS."""
        if use_openai:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
                response = client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=text,
                )
                response.stream_to_file(output_path)
                return
            except Exception as e:
                print(f"[video] OpenAI TTS failed ({e}), falling back to gTTS")

        from gtts import gTTS

        tts = gTTS(text=text, lang="en")
        tts.save(output_path)

    @staticmethod
    def make_clip(slide_image_path: str, audio_path: str, output_path: str) -> None:
        """Combine a slide PNG and an audio file into a video clip via moviepy."""
        from moviepy.editor import AudioFileClip, ImageClip

        audio = AudioFileClip(audio_path)
        clip = ImageClip(slide_image_path).set_duration(audio.duration).set_audio(audio)
        clip.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )
        audio.close()
        clip.close()

    @staticmethod
    def concat_clips(clip_folder: str, output_path: str) -> None:
        """Concatenate all .mp4 clips in clip_folder into output_path via ffmpeg."""
        clips = sorted(glob.glob(os.path.join(clip_folder, "*.mp4")))
        if not clips:
            raise RuntimeError(f"No .mp4 clips found in {clip_folder}")

        list_file = os.path.join(clip_folder, "concat_list.txt")
        with open(list_file, "w") as f:
            for clip in clips:
                f.write(f"file '{os.path.abspath(clip)}'\n")

        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_file,
                "-c", "copy",
                output_path,
            ],
            check=True,
            capture_output=True,
        )


class VideoGenerator:
    """Orchestrates the full PDF + script.md → MP4 pipeline per chapter."""

    def __init__(self, use_openai_tts: bool = True, voice: str = "alloy"):
        self.use_openai_tts = use_openai_tts
        self.voice = voice

    def generate(
        self,
        pdf_path: str,
        script_path: str,
        output_dir: str,
        video_name: str = "video",
    ) -> str:
        """Run the full pipeline for a single chapter.

        Returns the path to the generated MP4 file.
        """
        print(f"[video] Generating lecture video for {os.path.basename(output_dir)}")

        tmp_root = os.path.join(output_dir, "_video_tmp")
        slides_dir = os.path.join(tmp_root, "slide_images")
        audio_dir = os.path.join(tmp_root, "audio_files")
        clips_dir = os.path.join(tmp_root, "clips")
        os.makedirs(slides_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(clips_dir, exist_ok=True)

        try:
            slide_paths = VideoUtils.extract_slides(pdf_path, slides_dir, skip_first=True)
            narrations = VideoUtils.parse_script(script_path)

            count = min(len(slide_paths), len(narrations))
            if len(slide_paths) != len(narrations):
                print(
                    f"[video] Slide count ({len(slide_paths)}) vs script sections "
                    f"({len(narrations)}) differ — using first {count}"
                )

            for i in range(count):
                audio_path = os.path.join(audio_dir, f"audio_{i:03d}.mp3")
                clip_path = os.path.join(clips_dir, f"clip_{i:03d}.mp4")

                VideoUtils.make_audio(
                    narrations[i],
                    audio_path,
                    use_openai=self.use_openai_tts,
                    voice=self.voice,
                )
                VideoUtils.make_clip(slide_paths[i], audio_path, clip_path)

            output_path = os.path.join(output_dir, f"{video_name}.mp4")
            VideoUtils.concat_clips(clips_dir, output_path)
            print(f"[video] Saved: {output_path}")
            return output_path

        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def convert_directory(self, directory: str) -> List[str]:
        """Batch-generate videos for all chapter_*/ directories that have
        slides.pdf and script.md. Mirrors LaTeXToPPTXConverter.convert_directory().
        """
        chapter_dirs = sorted(glob.glob(os.path.join(directory, "chapter_*")))
        results = []
        for chapter_dir in chapter_dirs:
            pdf_path = os.path.join(chapter_dir, "slides.pdf")
            script_path = os.path.join(chapter_dir, "script.md")
            if not os.path.isfile(pdf_path) or not os.path.isfile(script_path):
                continue
            try:
                out = self.generate(pdf_path, script_path, chapter_dir)
                results.append(out)
            except Exception as e:
                print(f"[video] Failed for {chapter_dir}: {e}")
        return results
