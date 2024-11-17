# extractors.py
import requests
from typing import Dict, Any
from pytubefix import YouTube

class YouTubeExtractor:
    @staticmethod
    def get_video_id(url: str) -> str:
        """YouTube URL에서 비디오 ID를 추출합니다."""
        if "youtu.be" in url:
            return url.split("/")[-1]
        elif "youtube.com" in url:
            return url.split("v=")[1].split("&")[0]
        raise ValueError("잘못된 YouTube URL 형식입니다.")

    @staticmethod
    def get_video_info_noembed(url: str) -> Dict[str, Any]:
        """Noembed 서비스를 통해 비디오 정보를 가져옵니다."""
        try:
            noembed_url = f"https://noembed.com/embed?url={url}"
            response = requests.get(noembed_url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Noembed 정보 추출 실패: {str(e)}")

    @staticmethod
    def get_video_info_pytube(url: str) -> Dict[str, Any]:
        """PyTube를 사용하여 비디오 정보를 가져옵니다."""
        try:
            yt = YouTube(url)
            return {
                'title': yt.title,
                'author': yt.author,
                'length': yt.length,
                'views': yt.views,
                'description': yt.description,
                'thumbnail_url': yt.thumbnail_url,
                'publish_date': str(yt.publish_date) if yt.publish_date else None
            }
        except Exception as e:
            raise Exception(f"PyTube 정보 추출 실패: {str(e)}")

    @staticmethod
    def get_audio(url: str) -> str:
        """PyTube를 사용하여 오디오 스트림을 다운로드합니다."""
        try:
            yt = YouTube(url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            if audio_stream:
                audio_file = audio_stream.download(filename="temp_audio.mp4")
                return audio_file
            else:
                raise Exception("오디오 스트림을 찾을 수 없습니다.")
        except Exception as e:
            raise Exception(f"오디오 스트림 다운로드 실패: {str(e)}")

    @staticmethod
    def get_video_info(url: str) -> Dict[str, Any]:
        """PyTube 및 Noembed를 통합하여 비디오 정보를 가져옵니다."""
        try:
            return YouTubeExtractor.get_video_info_pytube(url)
        except Exception as pytube_error:
            print(f"PyTube 정보 추출 실패: {pytube_error}")
            try:
                return YouTubeExtractor.get_video_info_noembed(url)
            except Exception as noembed_error:
                raise Exception(f"비디오 정보 추출 실패: {pytube_error}; {noembed_error}")

    @staticmethod
    def get_transcription(url: str, model) -> Dict[str, Any]:
        """영상에서 자막을 추출하고 타임스탬프와 함께 반환합니다."""
        try:
            audio_file = YouTubeExtractor.get_audio(url)
            result = model.transcribe(audio_file)
            return result
        except Exception as e:
            raise Exception(f"자막 추출 실패: {str(e)}")