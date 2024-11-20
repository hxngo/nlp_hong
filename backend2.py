import os
import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import whisper
from pytubefix import YouTube
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ContentAnalyzer 클래스 추가
class ContentAnalyzer:
    def __init__(self):
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device="cpu"
        )
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english'
        )
        self.user_history = []
        try:
            self.load_history()
        except Exception as e:
            print(f"시청 기록 불러오기 실패: {str(e)}")
            self.user_history = []

    def summarize_content(self, text: str, max_length: int = 300) -> Dict[str, Any]:
        """영상 내용을 자동으로 요약합니다."""
        try:
            if not text or len(text.strip()) == 0:
                raise ValueError("텍스트가 비어있습니다.")

            original_length = len(text.split())
            if original_length < max_length:
                return {
                    'summary': text,
                    'summary_length': original_length,
                    'original_length': original_length
                }

            # 청크 크기를 더 작게 조정
            chunks = self._split_text(text, max_length=2000)
            summaries = []
        
            for chunk in chunks:
                if not chunk.strip():
                    continue
                
                try:
                    chunk_words = len(chunk.split())
                    min_length = min(100, chunk_words // 2)
                    max_chunk_length = min(max_length, chunk_words)
                
                    if max_chunk_length <= min_length:
                        summaries.append(chunk)
                        continue

                    summary = self.summarizer(
                        chunk,
                        max_length=max_chunk_length,
                        min_length=min_length,
                        do_sample=False,
                        num_beams=4,
                        length_penalty=2.0,
                        early_stopping=True
                    )[0]['summary_text']
                    summaries.append(summary)
                except Exception as e:
                    print(f"청크 요약 중 오류 발생: {str(e)}")
                    summaries.append(chunk)
        
            if not summaries:
                return {
                    'summary': text[:max_length],
                    'summary_length': min(len(text.split()), max_length),
                    'original_length': original_length
                }
        
            final_summary = " ".join(summaries)
        
            # 최종 요약이 너무 길면 다시 요약
            if len(final_summary.split()) > max_length:
                try:
                    final_summary = self.summarizer(
                        final_summary,
                        max_length=max_length,
                        min_length=min(100, len(final_summary.split()) // 2),
                        do_sample=False,
                        num_beams=4,
                        length_penalty=2.0,
                        early_stopping=True
                    )[0]['summary_text']
                except Exception as e:
                    print(f"최종 요약 생성 중 오류 발생: {str(e)}")
                    words = final_summary.split()[:max_length]
                    final_summary = " ".join(words)
        
            return {
                'summary': final_summary,
                'summary_length': len(final_summary.split()),
                'original_length': original_length
            }
        except Exception as e:
            print(f"요약 처리 중 오류 발생: {str(e)}")
            return {
                'summary': text[:max_length],
                'summary_length': min(len(text.split()), max_length),
                'original_length': original_length
            }

            

    def _split_text(self, text: str, max_length: int = 4000) -> List[str]:
        """긴 텍스트를 처리 가능한 크기의 청크로 나눕니다."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1
            if current_length > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def save_history(self) -> None:
        """시청 기록을 파일에 저장합니다."""
        try:
            with open('user_history.json', 'w', encoding='utf-8') as f:
                json.dump(self.user_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"시청 기록 저장 실패: {str(e)}")

    def load_history(self) -> None:
        """저장된 시청 기록을 불러옵니다."""
        try:
            if os.path.exists('user_history.json'):
                with open('user_history.json', 'r', encoding='utf-8') as f:
                    self.user_history = json.load(f)
        except Exception as e:
            print(f"시청 기록 불러오기 실패: {str(e)}")
            self.user_history = []

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
        except Exception as e:
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

class ContentAnalyzer:
    def get_content_recommendations(self, current_content: str, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """현재 컨텐츠와 유사한 이전 시청 기록을 추천합니다."""
        try:
            if not hasattr(self, 'user_history') or not self.user_history:
                return []
            
            all_contents = [current_content] + [item['content'] for item in self.user_history]
            tfidf_matrix = self.vectorizer.fit_transform(all_contents)
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            similar_indices = cosine_similarities.argsort()[::-1]
            recommendations = []
            
            for idx in similar_indices[:n_recommendations]:
                history_item = self.user_history[idx]
                recommendations.append({
                    'video_id': history_item.get('video_id', ''),
                    'title': history_item.get('title', ''),
                    'similarity_score': float(cosine_similarities[idx]),
                    'timestamp': history_item.get('timestamp', ''),
                    'metadata': history_item.get('metadata', {})
                })
            
            return recommendations
        except Exception as e:
            print(f"추천 컨텐츠 생성 실패: {str(e)}")
            return []
    
    def summarize_related_segments(self, segments: List[Dict[str, Any]], max_length: int = 150) -> List[Dict[str, str]]:
        """검색된 구간을 GPT 모델로 요약합니다."""
        try:
            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0,
                openai_api_key=self.openai_api_key
            )

            summarized_segments = []
            for segment in segments:
                prompt = f"다음 자막 내용을 {max_length} 단어 이내로 요약해 주세요:\n\n{segment['text']}\n"
                summary = llm.predict(prompt)
                summarized_segments.append({
                    'start_time': segment['start'],
                    'end_time': segment['end'],
                    'summary': summary
                })
            return summarized_segments
        except Exception as e:
            print(f"자막 요약 중 오류 발생: {str(e)}")
            return [{'start_time': segment['start'], 'end_time': segment['end'], 'summary': segment['text'][:max_length]} for segment in segments]

class VideoProcessor:
        
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수를 설정해주세요.")
        
        self.model = whisper.load_model("base")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )
        self.youtube_extractor = YouTubeExtractor()
        
        self.content_analyzer = ContentAnalyzer()
        

    def extract_key_points(self, text: str) -> Dict[str, Any]:
        """영상의 핵심 내용을 추출하고 구조화합니다."""
        try:
            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0,
                openai_api_key=self.openai_api_key
            )
        
            prompt = f"""
            다음 영상 내용에서 핵심 내용을 추출하여 구조화된 형식으로 정리해주세요:
        
            텍스트: {text}
        
            다음 형식으로 출력해주세요:
            1. 주제: [프레임워크/기술명]
            - 정의: 간단한 설명
            - 특징: 주요 특징 3-4개
            - 장단점: 장점과 단점 분석
            - 사용 대상: 적합한 사용자 프로필

            2. 핵심 용어:
            - [용어1]: 설명
            - [용어2]: 설명

            3. 결론:
            - 요약된 핵심 메시지
            """
            structured_summary = llm.predict(prompt)
            return {
                'structured_summary': structured_summary,
                'original_text': text
            }
        except Exception as e:
            raise Exception(f"핵심 내용 추출 실패: {str(e)}")
            
    def extract_keywords(self, text: str) -> Dict[str, Any]:
        """텍스트에서 주요 키워드를 추출하고 설명을 생성합니다."""
        try:
            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0,
                openai_api_key=self.openai_api_key
            )
        
            prompt = f"""
            다음 텍스트에서 주요 키워드를 추출하고 각각에 대해 자세히 설명해주세요:
        
            텍스트: {text}
        
            다음 형식으로 출력해주세요:
            1. [키워드1]
            - 설명: (키워드에 대한 자세한 설명)
            - 영상에서의 맥락: (영상에서 이 키워드가 어떤 맥락에서 사용되었는지)
        
            2. [키워드2]
            - 설명: (키워드에 대한 자세한 설명)
            - 영상에서의 맥락: (영상에서 이 키워드가 어떤 맥락에서 사용되었는지)
            """
        
            keywords_analysis = llm.predict(prompt)
        
            return {
                'keywords_analysis': keywords_analysis,
                'original_text': text
            }
        except Exception as e:
            raise Exception(f"키워드 추출 실패: {str(e)}")

class VideoProcessor:
    def __init__(self):
        self.user_history = []  # 시청 기록 초기화
        self.vectorizer = TfidfVectorizer()  # TF-IDF 초기화
        # 필요 시 추가 초기화 코드

    def _extract_transcription(self, url: str) -> Dict[str, Any]:
        """영상에서 자막을 추출하고 타임스탬프와 함께 반환합니다."""
        try:
            yt = YouTube(url)
            segments = []
            transcript = None

            # 자막 선택 및 처리 코드...
            # 자막 선택 코드 추가 및 처리
            available_captions = yt.captions
            print("Available captions:", available_captions.keys())
            caption_langs = ['ko', 'en', 'a.ko', 'a.en']

            for lang in caption_langs:
                if lang in available_captions:
                    transcript = available_captions[lang]
                    print(f"Selected caption language: {lang}")
                    break

            if transcript:
                caption_tracks = transcript.generate_srt_captions()
                for segment in caption_tracks.split('\n\n'):
                    if segment.strip():
                        lines = segment.split('\n')
                        if len(lines) >= 3:
                            try:
                                times = lines[1].split(' --> ')
                                start_time = self._time_to_seconds(times[0])
                                end_time = self._time_to_seconds(times[1])
                                text = ' '.join(lines[2:])
                                segments.append({'start': start_time, 'end': end_time, 'text': text})
                            except Exception as e:
                                print(f"Error parsing segment: {e}")
                                continue

            if not segments:
                print("No captions found, using Whisper...")
                audio = yt.streams.filter(only_audio=True).first()
                audio_file = audio.download(filename="temp_audio")
                result = self.model.transcribe(audio_file)
                segments = [{'start': seg['start'], 'end': seg['end'], 'text': seg['text']} for seg in result['segments']]
                if os.path.exists(audio_file):
                    os.remove(audio_file)

            if not segments:
                raise Exception("자막을 추출할 수 없습니다.")

            return {'text': ' '.join(seg['text'] for seg in segments), 'segments': segments}
        except Exception as e:
            print(f"자막 추출 중 오류 발생: {str(e)}")
            raise Exception(f"자막 추출 실패: {str(e)}")


    def summarize_segments(self, segments: List[Dict[str, Any]], video_length: int) -> List[Dict[str, Any]]:
        """영상의 구간을 길이에 따라 요약합니다.
    
        Args:
            segments: 자막 세그먼트 리스트
            video_length: 영상 길이(초)
        
        Returns:
            요약된 세그먼트 리스트
        """

        # segments가 비어있거나 문자열인 경우 처리
        if not segments or isinstance(segments, str):
            return []

        interval = 60 if video_length <= 600 else 180  # 1분(60초) 또는 3분(180초) 단위
        summarized_segments = []
        current_segment = []
        current_start = segments[0].get('start', 0)

        try:
            for segment in segments:
                # 세그먼트가 딕셔너리이고 필요한 키를 포함하는지 확인
                if not isinstance(segment, dict) or not all(key in segment for key in ['start', 'end', 'text']):
                    continue
                
                current_segment.append(segment['text'])

                if segment['end'] - current_start >= interval or segment == segments[-1]:

                    summarized_segments.append({
                        'start': current_start,
                        'end': segment['end'],
                        'text': ' '.join(current_segment)
                    })
                    current_segment = []
                    current_start = segment['end']

        except Exception as e:
            print(f"세그먼트 요약 중 오류 발생: {str(e)}")
        return []

        return summarized_segments

    def get_content_recommendations(self, current_content: str, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """현재 컨텐츠와 유사한 이전 시청 기록을 추천합니다."""
        try:
            if not self.user_history:
                # 시청 기록이 없으면 빈 리스트 반환
                return []

            all_contents = [current_content] + [item.get('content', '') for item in self.user_history]
            tfidf_matrix = self.vectorizer.fit_transform(all_contents)
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            similar_indices = cosine_similarities.argsort()[::-1]
            recommendations = []

            for idx in similar_indices[:n_recommendations]:
                if idx < len(self.user_history):
                    history_item = self.user_history[idx]
                    recommendations.append({
                        'video_id': history_item.get('video_id', ''),
                        'title': history_item.get('title', ''),
                        'similarity_score': round(float(cosine_similarities[idx]), 2),
                        'timestamp': history_item.get('timestamp', ''),
                        'metadata': history_item.get('metadata', {})
                    })

            return recommendations
        except Exception as e:
            print(f"추천 생성 실패: {str(e)}")
            return []

    def process_video(self, url: str) -> Dict[str, Any]:
        """비디오를 처리하고 관련된 정보 및 자막을 반환."""
        try:
            video_info = self._get_comprehensive_video_info(url)
            transcription = self._extract_transcription(url)

            documents = self._create_documents(transcription, video_info)
            vectorstore = self._create_vectorstore(documents)

            recommendations = []
            if isinstance(transcription, dict) and 'text' in transcription:
                try:
                    recommendations = self.get_content_recommendations(transcription['text'], n_recommendations=5)
                except Exception as e:
                    print(f"추천 컨텐츠 생성 실패: {str(e)}")

            return {
                "video_info": video_info,
                "vectorstore": vectorstore,
                "transcription": transcription,
                "recommendations": recommendations
            }
        except Exception as e:
            raise RuntimeError(f"비디오 처리 중 오류 발생: {str(e)}")

    def _time_to_seconds(self, time_str: str) -> float:
        """SRT 형식의 시간을 초 단위로 변환."""
        try:
            time_str = time_str.strip().replace(',', '.')
            if '.' not in time_str:
                time_str += '.000'
            hours, minutes, seconds = time_str.split(':')
            seconds, milliseconds = seconds.split('.')
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + float(f"0.{milliseconds}")
        except Exception as e:
            print(f"시간 변환 중 오류 발생: {str(e)}")
            return 0.0

class TranscriptManager:
    def __init__(self):
        self.transcripts = []
        self.load_transcripts()

    def add_transcript(self, video_id: str, segments: List[Dict[str, Any]]) -> None:
        """타임스탬프가 포함된 자막을 저장합니다."""
        transcript_data = {
            'video_id': video_id,
            'segments': segments,
            'created_at': datetime.now().isoformat()
        }
        self.transcripts.append(transcript_data)
        self.save_transcripts()

    def get_transcript(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        """특정 비디오의 자막을 가져옵니다."""
        for transcript in self.transcripts:
            if transcript['video_id'] == video_id:
                return transcript['segments']
        return None

    def search_in_transcript(self, video_id: str, query: str) -> List[Dict[str, Any]]:
        """자막에서 특정 키워드가 포함된 구간을 검색합니다."""
        transcript = self.get_transcript(video_id)
        if not transcript:
            return []

        results = []
        for segment in transcript:
            if query.lower() in segment['text'].lower():
                results.append({
                    'start_time': segment['start'],
                    'end_time': segment['end'],
                    'text': segment['text']
                })
        return results

    def save_transcripts(self) -> None:
        """자막 데이터를 파일에 저장합니다."""
        try:
            with open('transcripts.json', 'w', encoding='utf-8') as f:
                json.dump(self.transcripts, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"자막 저장 실패: {str(e)}")

    def load_transcripts(self) -> None:
        """저장된 자막 데이터를 불러옵니다."""
        try:
            if os.path.exists('transcripts.json'):
                with open('transcripts.json', 'r', encoding='utf-8') as f:
                    self.transcripts = json.load(f)
        except Exception as e:
            print(f"자막 불러오기 실패: {str(e)}")
            self.transcripts = []

class NoteManager:
    def __init__(self):
        self.notes = []
        self.load_notes()

    def add_note(self, content: str, video_info: Optional[Dict[str, Any]] = None) -> None:
        """메모를 추가합니다."""
        note = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'content': content,
            'video_info': video_info or {},
            'created_at': datetime.now().isoformat()
        }
        self.notes.append(note)
        self.save_notes()

    def get_notes(self) -> List[Dict[str, Any]]:
        """저장된 메모 목록을 반환합니다."""
        return sorted(self.notes, key=lambda x: x['created_at'], reverse=True)

    def remove_note(self, timestamp: str) -> None:
        """특정 메모를 삭제합니다."""
        self.notes = [n for n in self.notes if n['timestamp'] != timestamp]
        self.save_notes()

    def save_notes(self) -> None:
        
        try:
            with open('notes.json', 'w', encoding='utf-8') as f:
                json.dump(self.notes, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"메모 저장 실패: {str(e)}")

    def load_notes(self) -> None:
        """저장된 메모를 불러옵니다."""
        try:
            if os.path.exists('notes.json'):
                with open('notes.json', 'r', encoding='utf-8') as f:
                    self.notes = json.load(f)
        except Exception as e:
            print(f"메모 불러오기 실패: {str(e)}")
            self.notes = []

class BookmarkManager:
    def __init__(self):
        self.bookmarks = []
        self.load_bookmarks()

    def add_bookmark(self, timestamp: str, content: str, video_info: Optional[Dict[str, Any]] = None) -> None:
        """북마크를 추가합니다."""
        bookmark = {
            'timestamp': timestamp,
            'content': content,
            'video_info': video_info or {}
        }
        self.bookmarks.append(bookmark)
        self.save_bookmarks()

    def get_bookmarks(self) -> List[Dict[str, Any]]:
        """저장된 북마크 목록을 반환합니다."""
        return self.bookmarks

    def remove_bookmark(self, timestamp: str) -> None:
        """특정 북마크를 삭제합니다."""
        self.bookmarks = [b for b in self.bookmarks if b['timestamp'] != timestamp]
        self.save_bookmarks()

    def save_bookmarks(self) -> None:
        """북마크를 파일에 저장합니다."""
        try:
            with open('bookmarks.json', 'w', encoding='utf-8') as f:
                json.dump(self.bookmarks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"북마크 저장 실패: {str(e)}")

    def load_bookmarks(self) -> None:
        """저장된 북마크를 불러옵니다."""
        try:
            if os.path.exists('bookmarks.json'):
                with open('bookmarks.json', 'r', encoding='utf-8') as f:
                    self.bookmarks = json.load(f)
        except Exception as e:
            print(f"북마크 불러오기 실패: {str(e)}")
            self.bookmarks = []
__all__ = ["VideoProcessor", "BookmarkManager", "NoteManager", "TranscriptManager", "YouTubeExtractor", "ContentAnalyzer"]