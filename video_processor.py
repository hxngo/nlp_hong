import os
from typing import List, Dict, Any
import whisper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from extractors import YouTubeExtractor
from transformers import pipeline

class VideoProcessor:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수를 설정해주세요.")

        # Whisper 모델 로드
        try:
            self.model = whisper.load_model("base", device="cpu")
        except Exception as e:
            print(f"Whisper 모델 로드 중 오류 발생: {e}")
            raise

        # 임베딩 모델 로드
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            print(f"임베딩 모델 로드 중 오류 발생: {e}")
            raise

        # 요약 모델 로드
        try:
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device="cpu"
            )
        except Exception as e:
            print(f"요약 모델 로드 중 오류 발생: {e}")
            raise

        # 텍스트 스플리터 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # YouTube 추출기 초기화
        self.youtube_extractor = YouTubeExtractor()

    def process_video(self, url: str) -> Dict[str, Any]:
        """YouTube 영상을 처리하고 필요한 정보를 추출합니다."""
        try:
            # 비디오 정보 추출
            video_info = self._get_comprehensive_video_info(url)
            
            # 자막 추출
            transcription = self._extract_transcription(url)
            
            # 요약 생성
            if isinstance(transcription, dict) and 'text' in transcription:
                summary = self._generate_summary(transcription['text'])
            else:
                summary = self._generate_summary(transcription)
            
            # 문서 생성 및 벡터스토어 생성
            documents = self._create_documents(transcription, video_info)
            vectorstore = self._create_vectorstore(documents)
            
            return {
                "video_info": video_info,
                "vectorstore": vectorstore,
                "transcription": transcription,
                "summary": summary
            }
        except Exception as e:
            raise RuntimeError(f"비디오 처리 중 오류 발생: {e}")

    def _get_comprehensive_video_info(self, url: str) -> Dict[str, Any]:
        """여러 방법을 통해 종합적인 비디오 정보를 수집합니다."""
        video_info = {}
        errors = []
        
        try:
            pytube_info = self.youtube_extractor.get_video_info_pytube(url)
            video_info.update(pytube_info)
        except Exception as e:
            errors.append(f"PyTube 정보 수집 실패: {e}")
            
        try:
            noembed_info = self.youtube_extractor.get_video_info_noembed(url)
            for key, value in noembed_info.items():
                if key not in video_info or not video_info[key]:
                    video_info[key] = value
        except Exception as e:
            errors.append(f"Noembed 정보 수집 실패: {e}")
            
        if not video_info:
            raise Exception(f"비디오 정보를 가져오는데 실패했습니다. 오류: {'; '.join(errors)}")
            
        return video_info

    def _extract_transcription(self, url: str) -> Dict[str, Any]:
        """영상에서 자막을 추출합니다."""
        try:
            audio_file = self.youtube_extractor.get_audio(url)
            result = self.model.transcribe(audio_file)
            
            if os.path.exists(audio_file):
                os.remove(audio_file)
                
            return result
        except Exception as e:
            raise Exception(f"자막 추출 실패: {str(e)}")

    def _generate_summary(self, text: str, max_length: int = 300) -> Dict[str, Any]:
        """텍스트 요약을 생성합니다."""
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

            chunks = self._split_text(text, max_length=4000)
            summaries = []
            
            for chunk in chunks:
                if not chunk.strip():
                    continue
                    
                try:
                    summary = self.summarizer(
                        chunk,
                        max_length=min(max_length, len(chunk.split())),
                        min_length=min(100, len(chunk.split()) // 2),
                        do_sample=False
                    )[0]['summary_text']
                    summaries.append(summary)
                except Exception as e:
                    print(f"청크 요약 중 오류 발생: {str(e)}")
                    continue
            
            if not summaries:
                raise ValueError("요약을 생성할 수 없습니다.")
            
            final_summary = " ".join(summaries)
            
            if len(final_summary.split()) > max_length:
                try:
                    final_summary = self.summarizer(
                        final_summary,
                        max_length=max_length,
                        min_length=min(100, len(final_summary.split()) // 2),
                        do_sample=False
                    )[0]['summary_text']
                except Exception as e:
                    print(f"최종 요약 생성 중 오류 발생: {str(e)}")
            
            return {
                'summary': final_summary,
                'summary_length': len(final_summary.split()),
                'original_length': original_length
            }
        except Exception as e:
            raise Exception(f"요약 생성 실패: {str(e)}")

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

    def _create_documents(self, transcription: Dict[str, Any], video_info: Dict[str, Any]) -> List[Document]:
        """텍스트와 비디오 정보를 문서 형태로 변환합니다."""
        try:
            metadata = {
                'title': video_info.get('title', ''),
                'author': video_info.get('author', ''),
                'length': video_info.get('length', 0),
                'source_type': 'youtube_video'
            }
            
            text = transcription['text'] if isinstance(transcription, dict) else transcription
            documents = [Document(page_content=text, metadata=metadata)]
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            raise Exception(f"문서 생성 실패: {str(e)}")

    def _create_vectorstore(self, documents: List[Document]) -> Chroma:
        """문서로부터 벡터스토어를 생성합니다."""
        try:
            return Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="video_db"
            )
        except Exception as e:
            raise Exception(f"벡터스토어 생성 실패: {str(e)}")

    def search_content(self, vectorstore: Chroma, query: str) -> Dict[str, Any]:
        """벡터스토어에서 쿼리에 관련된 내용을 검색합니다."""
        try:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=self.openai_api_key
            )
            
            prompt_template = """
            다음 영상 내용을 바탕으로 질문에 답변해주세요:
            {context}
            질문: {question}
            답변:
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            result = qa_chain({"query": query})
            
            return {
                'answer': result['result'],
                'source_documents': [
                    {
                        'content': doc.page_content,
                        'metadata': doc.metadata
                    } for doc in result['source_documents']
                ]
            }
        except Exception as e:
            raise Exception(f"검색 실패: {str(e)}")