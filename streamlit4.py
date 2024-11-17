import streamlit as st
import os
from backend import VideoProcessor, BookmarkManager, NoteManager, TranscriptManager, YouTubeExtractor
from datetime import datetime
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="YouTube 강의 검색 도우미",
    page_icon="🎓",
    layout="wide"
)

# 스타일 설정
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .css-1v0mbdj.ebxwdo61 {
        width: 100%;
        max-width: none;
    }
    .stTextArea textarea {
        font-size: 1rem;
        line-height: 1.5;
    }
    .memo-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .memo-metadata {
        color: #666;
        font-size: 0.8rem;
    }
    .transcript-segment {
        padding: 5px 0;
        border-bottom: 1px solid #eee;
    }
    .timestamp {
        color: #666;
        font-size: 0.9rem;
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# 세션 상태 초기화
if 'processor' not in st.session_state:
    st.session_state.processor = VideoProcessor()
if 'bookmark_manager' not in st.session_state:
    st.session_state.bookmark_manager = BookmarkManager()
if 'note_manager' not in st.session_state:
    st.session_state.note_manager = NoteManager()
if 'transcript_manager' not in st.session_state:
    st.session_state.transcript_manager = TranscriptManager()
if 'current_video' not in st.session_state:
    st.session_state.current_video = None
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

def process_video():
    """비디오 처리 함수"""
    try:
        with st.spinner('영상을 처리하는 중입니다...'):
            result = st.session_state.processor.process_video(st.session_state.youtube_url)
            
            # 자막 정보 저장
            video_id = YouTubeExtractor.get_video_id(st.session_state.youtube_url)
            if isinstance(result['transcription'], dict) and 'segments' in result['transcription']:
                st.session_state.transcript_manager.add_transcript(
                    video_id,
                    result['transcription']['segments']
                )
                result['transcription'] = result['transcription']['text']
            
            st.session_state.current_video = result
            st.success('영상 처리가 완료되었습니다!')
            return True
    except Exception as e:
        st.error(f'영상 처리 중 오류가 발생했습니다: {str(e)}')
        return False

def search_content():
    """컨텐츠 검색 함수"""
    try:
        if not st.session_state.search_query:
            st.warning('검색어를 입력해주세요.')
            return
        
        with st.spinner('검색 중...'):
            result = st.session_state.processor.search_content(
                st.session_state.current_video['vectorstore'],
                st.session_state.search_query
            )
            
            # 검색 기록 저장
            st.session_state.search_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'query': st.session_state.search_query,
                'answer': result['answer']
            })
            
            return result
    except Exception as e:
        st.error(f'검색 중 오류가 발생했습니다: {str(e)}')
        return None

def save_note(note_content: str):
    """메모 저장 함수"""
    if note_content.strip():
        video_info = None
        if st.session_state.current_video:
            video_info = st.session_state.current_video['video_info']
        
        st.session_state.note_manager.add_note(
            content=note_content,
            video_info=video_info
        )
        return True
    return False

def format_time(seconds: float) -> str:
    """초를 mm:ss 형식으로 변환"""
    minutes = int(seconds) // 60
    seconds = int(seconds) % 60
    return f"{minutes:02d}:{seconds:02d}"

def main():
    st.title('🎓 YouTube 강의 검색 도우미')
    
    # 좌우 컬럼 분할
    left_col, right_col = st.columns([1, 3])
    
    # 왼쪽 컬럼 (메모 및 설정)
    with left_col:
        st.header('📝 메모 및 설정')
        
        # YouTube URL 입력
        youtube_url = st.text_input(
            'YouTube URL을 입력하세요',
            key='youtube_url',
            placeholder='https://www.youtube.com/watch?v=...'
        )
        
        # 영상 처리 버튼
        if st.button('영상 처리 시작', key='process_button'):
            process_video()
        
        st.divider()
        
        # 메모 입력 섹션
        st.subheader('✏️ 메모하기')
        note_content = st.text_area(
            '메모를 입력하세요:',
            height=200,
            placeholder='여기에 메모를 작성하세요...'
        )
        
        if st.button('메모 저장', use_container_width=True):
            if save_note(note_content):
                st.success('메모가 저장되었습니다!')
                st.balloons()
            else:
                st.warning('메모를 작성해 주세요!')
        
        # 검색 섹션
        st.divider()
        st.subheader('🔍 검색')
        search_query = st.text_input(
            '검색어를 입력하세요',
            key='search_query',
            placeholder='궁금한 내용을 입력하세요'
        )
        
        if st.button('검색', key='search_button'):
            if st.session_state.current_video is None:
                st.warning('먼저 영상을 처리해주세요.')
            else:
                st.session_state.search_result = search_content()
    
    # 오른쪽 컬럼 (메인 컨텐츠)
    with right_col:
        # 비디오 플레이어
        if youtube_url:
            st.video(youtube_url)
            
            if st.session_state.current_video:
                with st.expander('📊 영상 정보'):
                    st.json(st.session_state.current_video['video_info'])
                
                # 자막 검색 섹션
                st.subheader('🔍 자막 검색')
                transcript_search = st.text_input(
                    '검색할 키워드를 입력하세요',
                    key='transcript_search',
                    placeholder='자막에서 검색할 키워드를 입력하세요...'
                )
                
                if transcript_search:
                    video_id = YouTubeExtractor.get_video_id(youtube_url)
                    search_results = st.session_state.transcript_manager.search_in_transcript(
                        video_id, 
                        transcript_search
                    )
                    
                    if search_results:
                        st.write(f"🎯 검색 결과: {len(search_results)}개 구간 발견")
                        for idx, result in enumerate(search_results):
                            with st.expander(f"구간 {idx + 1} ({format_time(result['start_time'])} ~ {format_time(result['end_time'])})"):
                                st.write(result['text'])
                                
                                # 타임스탬프 링크 생성
                                timestamp_url = f"{youtube_url}&t={int(result['start_time'])}"
                                st.markdown(f"🎥 [이 구간으로 이동]({timestamp_url})")
                                
                                # 북마크 추가 버튼
                                if st.button(f'이 구간 북마크 추가 #{idx}', key=f'segment_bookmark_{idx}'):
                                    st.session_state.bookmark_manager.add_bookmark(
                                        timestamp=format_time(result['start_time']),
                                        content=result['text'],
                                        video_info={
                                            'title': st.session_state.current_video['video_info'].get('title', ''),
                                            'url': youtube_url,
                                            'timestamp': int(result['start_time'])
                                        }
                                    )
                                    st.success('북마크가 추가되었습니다!')
                    else:
                        st.info('검색 결과가 없습니다.')

                # 전체 자막 보기
                with st.expander("📝 전체 자막 보기"):
                    video_id = YouTubeExtractor.get_video_id(youtube_url)
                    transcript = st.session_state.transcript_manager.get_transcript(video_id)
                    
                    if transcript:
                        for segment in transcript:
                            st.markdown(f"""
                            <div class="transcript-segment">
                                <span class="timestamp">[{format_time(segment['start'])}]</span> {segment['text']}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info('자막 정보가 없습니다.')
        
        # 검색 결과
        if 'search_result' in st.session_state and st.session_state.search_result:
            st.subheader('🔍 검색 결과')
            result = st.session_state.search_result
            
            st.markdown(f"**답변:**\n{result['answer']}")
            
            with st.expander('🔎 관련 구간'):
                for idx, doc in enumerate(result['source_documents']):
                    st.markdown(f"**구간 {idx+1}**")
                    st.write(doc['content'])
                    if st.button(f'북마크 추가 #{idx}', key=f'bookmark_{idx}'):
                        st.session_state.bookmark_manager.add_bookmark(
                            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            content=doc['content'],
                            video_info=doc['metadata']
                        )
                        st.success('북마크가 추가되었습니다!')
    
    # 북마크 섹션
    st.divider()
    st.header('📚 북마크')
    bookmarks = st.session_state.bookmark_manager.get_bookmarks()
    
    if bookmarks:
        for idx, bookmark in enumerate(bookmarks):
            with st.expander(f"📌 북마크 {idx+1} - {bookmark['timestamp']}"):
                st.write(bookmark['content'])
                
                # 비디오 URL과 타임스탬프가 있는 경우 링크 생성
                if 'video_info' in bookmark and 'url' in bookmark['video_info']:
                    video_url = bookmark['video_info']['url']
                    timestamp = bookmark['video_info'].get('timestamp', 0)
                    timestamp_url = f"{video_url}&t={timestamp}"
                    st.markdown(f"🎥 [이 구간으로 이동]({timestamp_url})")
                
                if st.button('삭제', key=f'delete_bookmark_{idx}'):
                    st.session_state.bookmark_manager.remove_bookmark(bookmark['timestamp'])
                    st.rerun()
    else:
        st.info('저장된 북마크가 없습니다.')
    
    # 검색 기록
    st.divider()
    st.header('📜 검색 기록')
    if st.session_state.search_history:
        history_df = pd.DataFrame(st.session_state.search_history)
        st.dataframe(
            history_df,
            column_config={
                'timestamp': '시간',
                'query': '검색어',
                'answer': '답변'
            },
            hide_index=True
        )
    else:
        st.info('검색 기록이 없습니다.')

if __name__ == "__main__":
    main()