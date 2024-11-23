import os
import json
import requests
import numpy as np
import logging
import streamlit as st
import whisper
from typing import List, Dict, Any, Optional
from datetime import datetime
from pytubefix import YouTube
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2 import service_account


# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_youtube_service():
    api_key = os.getenv('YOUTUBE_API_KEY')
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        return youtube
    except Exception as e:
        print(f"YouTube API 서비스 초기화 실패: {e}")
        return None

# ContentAnalyzer 클래스
class ContentAnalyzer:
    
    def __init__(self):
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device="cpu"
        )
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.user_history = []
        self.load_history()

    def add_to_history(self, video_data: Dict[str, Any]) -> None:
        try:
            history_item = {
                "video_id": video_data.get("video_id", ""),
                "title": video_data.get("title", ""),
                "content": video_data.get("content", ""),
                "timestamp": datetime.now().isoformat(),
                "metadata": video_data.get("metadata", {})
            }
            self.user_history.append(history_item)
            self.save_history()
        except Exception as e:
            logging.error(f"시청 기록 추가 실패: {e}")

    def get_video_recommendations(self, video_id: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """현재 비디오와 관련된 YouTube 동영상을 검색합니다."""
        try:
            from googleapiclient.discovery import build
            youtube = build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))
        
            # 현재 비디오의 상세 정보 가져오기
            video_response = youtube.videos().list(
                part='snippet,topicDetails',
                id=video_id
            ).execute()
        
            if not video_response['items']:
                return []
            
            current_video = video_response['items'][0]['snippet']
                
            # 검색 쿼리 생성 (제목, 설명, 태그 기반)
            search_query = f"{current_video['title']}"
            if 'tags' in current_video:
                search_query += f" {' '.join(video_info['tags'][:3])}"
            if 'description' in video_info:
                # 설명에서 첫 100자만 사용
                search_query += f" {video_info['description'][:100]}"
            
            # 연관 동영상 검색
            search_response = youtube.search().list(
                q=search_query,
                type='video',
                part='snippet',
                maxResults=max_results + 1,  # 여유있게 더 많은 결과 요청
                relevanceLanguage='ko',
                order='relevance',
                videoCategoryId=current_video.get('categoryId'),
                fields='items(id,snippet(title,description,thumbnails))'
            ).execute()
        
            # 현재 비디오 제외하고 결과 처리
            recommendations = []
            for item in search_response.get('items', []):
                if item['id']['videoId'] != video_id:
                    try:
                        video_detail = youtube.videos().list(
                            part='statistics,snippet',
                            id=item['id']['videoId']
                        ).execute()
                    
                        if video_detail['items']:
                            video_info = video_detail['items'][0]
                            recommendations.append({
                                'video_id': item['id']['videoId'],
                                'title': video_info['snippet']['title'],
                                'description': video_info['snippet']['description'],
                                'thumbnail': item['snippet']['thumbnails']['high']['url'],
                                'view_count': int(video_info['statistics'].get('viewCount', 0)),
                                'published_at': video_info['snippet']['publishedAt']
                            })
                    except Exception as detail_error:
                        print(f"비디오 상세 정보 가져오기 실패: {detail_error}")
                        continue
                
                    if len(recommendations) >= max_results:
                        break
                    
            return recommendations
        
        except Exception as e:
            print(f"YouTube API 검색 실패: {e}")
            return []

    def get_content_recommendations(self, current_content: str, current_video_id: Optional[str] = None, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """현재 컨텐츠와 유사한 이전 시청 기록을 추천합니다.
    
        Args:
            current_content (str): 현재 컨텐츠의 내용
            current_video_id (Optional[str]): 현재 비디오 ID (중복 추천 방지용)
            n_recommendations (int): 추천할 컨텐츠 개수
        
        Returns:
            List[Dict[str, Any]]: 추천된 컨텐츠 목록
        """
        try:
            # 시청 기록이 없으면 빈 리스트 반환
            if not self.user_history:
                return []

            # 현재 비디오를 제외한 시청 기록 필터링
            filtered_history = [
                item for item in self.user_history 
                if item["video_id"] != current_video_id
            ] if current_video_id else self.user_history

            if not filtered_history:
                return []

            # TF-IDF 매트릭스 생성 및 코사인 유사도 계산
            all_contents = [current_content] + [item["content"] for item in filtered_history]
            tfidf_matrix = self.vectorizer.fit_transform(all_contents)
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            # 유사도 점수를 기준으로 정렬된 추천 목록 생성
            recommendations = []
            for idx, similarity in enumerate(cosine_similarities):
                recommendations.append({
                    "video_id": filtered_history[idx]["video_id"],
                    "title": filtered_history[idx]["title"],
                    "similarity_score": float(similarity),
                    "timestamp": filtered_history[idx]["timestamp"],
                    "metadata": filtered_history[idx]["metadata"]
                })
        
            # 유사도 점수로 정렬하여 상위 n개 반환
            return sorted(
                recommendations,
                key=lambda x: x["similarity_score"],
                reverse=True
            )[:n_recommendations]

        except Exception as e:
            logging.error(f"추천 콘텐츠 생성 실패: {str(e)}")
            return []

    def summarize_with_gpt(self, text: str, max_length: int = 300) -> Dict[str, Any]:
        """GPT를 사용하여 영상 내용을 요약합니다."""
        try:
            if not text or len(text.strip()) == 0:
                raise ValueError("텍스트가 비어있습니다.")
            
            original_length = len(text.split())
        
            # GPT 모델 초기화
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        
            # 프롬프트 템플릿 설정
            prompt_template = """
            다음 영상 내용을 {max_length}단어 이내로 요약해주세요. 
            주요 내용과 핵심 포인트를 포함해야 합니다.

            영상 내용:
            {text}

            요약:
            """
        
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["text", "max_length"]
            )
        
            # GPT로 요약 생성
            chain = prompt | llm
            summary = chain.invoke({"text": text, "max_length": max_length})
        
            return {
                'summary': summary.content,
                'summary_length': len(summary.content.split()),
                'original_length': original_length
            }
        
        except Exception as e:
            raise Exception(f"GPT 요약 생성 실패: {str(e)}")


    def save_history(self) -> None:
        try:
            with open("user_history.json", "w", encoding="utf-8") as f:
                json.dump(self.user_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"시청 기록 저장 실패: {e}")

    def load_history(self) -> None:
        try:
            if os.path.exists("user_history.json"):
                with open("user_history.json", "r", encoding="utf-8") as f:
                    self.user_history = json.load(f)
        except Exception as e:
            logging.error(f"시청 기록 불러오기 실패: {e}")
            self.user_history = []

    def remove_from_history(self, video_id: str) -> None:
        """시청 기록에서 특정 항목을 삭제합니다."""
        try:
            self.user_history = [item for item in self.user_history if item['video_id'] != video_id]
            self.save_history()
        except Exception as e:
            logging.error(f"시청 기록 삭제 실패: {e}")


# YouTubeExtractor 클래스
class YouTubeExtractor:
    @staticmethod
    def get_video_id(url: str) -> str:
        if "youtu.be" in url:
            return url.split("/")[-1]
        elif "youtube.com" in url:
            return url.split("v=")[1].split("&")[0]
        raise ValueError("잘못된 YouTube URL 형식입니다.")

    @staticmethod
    def get_video_info_pytube(url: str) -> Dict[str, Any]:
        try:
            yt = YouTube(url)
            return {
                "title": yt.title,
                "author": yt.author,
                "length": yt.length,
                "views": yt.views,
                "description": yt.description,
                "thumbnail_url": yt.thumbnail_url,
                "publish_date": str(yt.publish_date) if yt.publish_date else None
            }
        except Exception as e:
            logging.error(f"PyTube 정보 추출 실패: {e}")
            return {}

# VideoProcessor 클래스
class VideoProcessor:

    # VideoProcessor 클래스의 __init__ 메서드 수정
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    
        if not self.openai_api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        if not self.youtube_api_key:
            raise ValueError("YouTube API 키가 설정되지 않았습니다.")

        self.model = whisper.load_model("base")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.youtube_extractor = YouTubeExtractor()
        self.content_analyzer = ContentAnalyzer()

    def summarize_segments(self, segments: List[Dict[str, Any]], video_length: int) -> List[Dict[str, Any]]:
        """영상의 구간을 길이에 따라 요약합니다.
    
        Args:
            segments: 자막 세그먼트 리스트
            video_length: 영상 길이(초)
        
        Returns:
            요약된 세그먼트 리스트
        """
        # segments가 비어있거나 유효하지 않을 경우 처리
        valid_segments = [s for s in segments if isinstance(s, dict) and all(k in s for k in ['start', 'end', 'text'])]
        if not valid_segments:
            return []

        # 영상 길이에 따른 구간 설정
        interval = 60 if video_length <= 600 else 180  # 1분(60초) 또는 3분(180초) 단위
        summarized_segments = []
        current_segment = []
        current_start = valid_segments[0]['start']

        try:
            for segment in valid_segments:
                current_segment.append(segment['text'])
                # interval 기준으로 요약하거나 마지막 세그먼트일 경우
                if segment['end'] - current_start >= interval or segment == valid_segments[-1]:
                    summarized_segments.append({
                        'start': current_start,
                        'end': segment['end'],
                        'text': ' '.join(current_segment)
                    })
                    current_segment = []
                    current_start = segment['end']
        except Exception as e:
            print(f"세그먼트 요약 중 오류 발생: {e}")
            return []

        return summarized_segments

    def _time_to_seconds(self, time_str: str) -> float:
        """SRT 형식의 시간을 초 단위로 변환합니다."""
        try:
            time_str = time_str.strip().replace(',', '.')
            if '.' not in time_str:
                time_str += '.000'
            hours, minutes, seconds = time_str.split(':')
            seconds, milliseconds = seconds.split('.')
            total_seconds = (
                int(hours) * 3600 + 
                int(minutes) * 60 + 
                int(seconds) + 
                float(f"0.{milliseconds}")
            )
            return total_seconds
        except Exception as e:
            print(f"시간 변환 중 오류 발생: {str(e)}")
            return 0.0


    def process_video(self, url: str) -> Dict[str, Any]:
        try:
            video_info = self.youtube_extractor.get_video_info_pytube(url)
            transcription = self._extract_transcription(url)
        
            # 자막 정보 저장
            video_id = self.youtube_extractor.get_video_id(url)
        
            # GPT 요약 및 추천 컨텐츠 초기화
            summary = None
            recommendations = []
        
            if isinstance(transcription, dict) and 'text' in transcription:
                # 자막 저장
                if 'segments' in transcription:
                    st.session_state.transcript_manager.add_transcript(
                        video_id,
                        transcription['segments']
                    )
            
                # GPT 요약 생성
                summary = self.content_analyzer.summarize_with_gpt(
                    transcription['text'],
                    max_length=300
                )
            
                # YouTube API를 사용한 추천 컨텐츠 생성
                try:
                    from googleapiclient.discovery import build
                    api_key = os.getenv('YOUTUBE_API_KEY')
                    if not api_key:
                        raise ValueError("YouTube API 키가 설정되지 않았습니다")
                    
                    youtube = build('youtube', 'v3', developerKey=api_key)
                
                    # 검색 쿼리 생성
                    search_query = video_info.get('title', '')
                
                    # 연관 동영상 검색
                    search_response = youtube.search().list(
                        q=search_query,
                        type='video',
                        part='snippet',
                        maxResults=10,
                        relevanceLanguage='ko',
                        order='relevance',
                        fields='items(id,snippet(title, description, thumbnails))'
                    ).execute()
                
                    # 현재 비디오 제외하고 결과 처리
                    recommendations = []
                    for item in search_response.get('items', []):
                        if item['id']['videoId'] != video_id:
                            recommendations.append({
                                'video_id': item['id']['videoId'],
                                'title': item['snippet']['title'],
                                'description': item['snippet']['description'], 
                                'thumbnail': item['snippet']['thumbnails']['high']['url']
                            })

                            if len(recommendations) >= 5:
                                break
                            
                except Exception as e:
                    print(f"YouTube API 검색 실패: {e}")
                    recommendations = []
            
            documents = self._create_documents(transcription, video_info)
            vectorstore = self._create_vectorstore(documents)

            return {
                "video_info": video_info,
                "vectorstore": vectorstore,
                "transcription": transcription,
                "summary": summary,
                "recommendations": recommendations
            }
        except Exception as e:
            raise RuntimeError(f"비디오 처리 중 오류 발생: {e}")
        
    def _extract_transcription(self, url: str) -> Dict[str, Any]:
        try:
            yt = YouTube(url)
            segments = []
            transcript = None
        
            # 자막 확인 및 선택
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
                    if not segment.strip():
                        continue
                
                    lines = segment.split('\n')
                    if len(lines) >= 3:
                        try:
                            times = lines[1].split(' --> ')
                            start_time = self._time_to_seconds(times[0])
                            end_time = self._time_to_seconds(times[1])
                            text = ' '.join(lines[2:])
                            segments.append({
                                'start': start_time,
                                'end': end_time,
                                'text': text
                            })
                        except Exception as e:
                            print(f"세그먼트 파싱 오류: {str(e)}")
                            continue
        
            # 자막이 없거나 세그먼트가 비어있는 경우 Whisper 사용
            if not segments:
                print("자막을 찾을 수 없어 Whisper 사용...")
                audio = yt.streams.filter(only_audio=True).first()
                audio_file = audio.download(filename="temp_audio.mp3")
                result = self.model.transcribe(audio_file)
                segments = [
                    {
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text']
                    }
                    for segment in result['segments']
                ]
                if os.path.exists(audio_file):
                    os.remove(audio_file)

            if not segments:
                raise Exception("자막을 추출할 수 없습니다.")

            return {
                'text': ' '.join(segment['text'] for segment in segments),
                'segments': segments
            }
        except Exception as e:
            logging.error(f"자막 추출 중 오류 발생: {e}")
            return {"text": "", "segments": []}

    def _create_documents(self, transcription: Dict[str, Any], video_info: Dict[str, Any]) -> List[Document]:
        metadata = {"title": video_info.get("title", ""), "author": video_info.get("author", "")}
        documents = [Document(page_content=transcription["text"], metadata=metadata)]
        return self.text_splitter.split_documents(documents)

    def _create_vectorstore(self, documents: List[Document]) -> Chroma:
        try:
            if not documents:
                raise ValueError("문서가 비어있습니다")
            
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="video_db"
            )
        
            if not vectorstore:
                raise ValueError("벡터스토어 생성 실패")
            
            return vectorstore
        except Exception as e:
            print(f"벡터스토어 생성 실패: {str(e)}")
            return None

    def search_content(self, vectorstore: Chroma, query: str, role: str = "일반") -> Dict[str, Any]:
        """벡터스토어에서 쿼리에 관련된 내용을 검색합니다.

        Args:
            vectorstore: chroma 벡터스토어 인스턴스
            query: 검색할 질문
            role: 답변 스타일 (기본값: "일반")

        Returns:
            검색 결과와 관련 문서를 포함한 딕셔너리

        """
        try:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.2,
                openai_api_key=self.openai_api_key
            )
            
            prompt_template = """
            다음 영상 내용을 바탕으로 질문에 답변해주세요.
            답변은 명확하고 이해하기 쉽게 작성해주세요.

            영상 내용:
            {context}

            질문: {question}

            답변:
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # QA 체인 생성
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_kwargs={"k": 3} #상위 3개 문서만 검색
                ),
                chain_type_kwargs={
                    "prompt": PROMPT,
                    "verbose": False #디버그 출력 비활성화
                },
                return_source_documents=True
            )

            # 검색 실행
            result = qa_chain.invoke({"query": query})

            # 결과 반환
            return {
                'answer': result['result'],
                'source_documents': [
                    {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'relevance_score': getattr(doc, 'relevance_score', None)
                    } for doc in result['source_documents']
                ]
            }
            
        except Exception as e:
            raise Exception(f"검색 실패: {str(e)}")

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
        try:
            for transcript in self.transcripts:
                if transcript.get('video_id') == video_id:
                    return transcript.get('segments', [])
            return None
        except Exception as e:
            print(f"자막 가져오기 실패: {str(e)}")
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
                    content = f.read().strip()  # 파일 내용 읽기
                    if content:  # 내용이 있는 경우에만 파싱
                        self.transcripts = json.loads(content)
                    else:
                        # 빈 파일인 경우 빈 리스트로 초기화
                        self.transcripts = []
                        # 빈 리스트로 파일 초기화
                        with open('transcripts.json', 'w', encoding='utf-8') as f:
                            json.dump([], f, ensure_ascii=False)
            else:
                # 파일이 없는 경우 새로 생성
                self.transcripts = []
                with open('transcripts.json', 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False)
        except Exception as e:
            logging.error(f"자막 불러오기 실패: {str(e)}")
            self.transcripts = []


class NoteManager:
    def __init__(self):
        self.notes = []
        self.load_notes()

    def save_note(self, note_content: str, video_info: Optional[Dict[str, Any]] = None) -> bool:
        """메모를 저장합니다."""
        if note_content.strip():
            self.add_note(
                content=note_content,
                video_info=video_info
            )
            return True
        return False
    
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
        """메모를 파일에 저장합니다."""
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