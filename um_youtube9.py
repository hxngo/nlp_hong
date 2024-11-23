import streamlit as st
import os
from datetime import datetime
import pandas as pd
from backend_youtube6 import (
    VideoProcessor, 
    BookmarkManager, 
    NoteManager, 
    TranscriptManager, 
    YouTubeExtractor
)
from bookmark_sidebar import show_bookmark_sidebar
from search_history_sidebar import show_search_history_sidebar
from streamlit_extras.let_it_rain import rain

# 환경 변수 설정
if 'OPENAI_API_KEY' not in os.environ:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# 페이지 설정
st.set_page_config(
    page_title="YouTube 강의 검색 도우미",
    page_icon="🎓",
    layout="wide"
)

# 세션 상태 초기화
def initialize_session_state():
    """세션 상태를 초기화합니다."""
    try:
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
    except Exception as e:
        st.error(f"세션 상태 초기화 실패: {str(e)}")

def format_time(seconds: float) -> str:
    """초를 mm:ss 형식으로 변환"""
    minutes = int(seconds) // 60
    seconds = int(seconds) % 60
    return f"{minutes:02d}:{seconds:02d}"

def search_content():
    """컨텐츠 검색 함수"""
    try:
        if not st.session_state.search_query:
            st.warning('검색어를 입력해주세요.')
            return None

        if st.session_state.current_video is None:
            st.warning('먼저 영상을 처리해주세요.')
            return None

        with st.spinner('검색 중...'):
            result = st.session_state.processor.search_content(
                st.session_state.current_video['vectorstore'],
                st.session_state.search_query
            )
            st.session_state.search_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'query': st.session_state.search_query,
                'answer': result['answer']
            })
            return result
    except Exception as e:
        st.error(f'검색 중 오류가 발생했습니다: {str(e)}')
        return None

# 메인 애플리케이션
def main():
    initialize_session_state()
    processor = st.session_state.processor

    st.title('🎓 YouTube 강의 검색 도우미')

    # 좌우 컬럼
    left_col, right_col = st.columns([1, 3])

    # 왼쪽 컬럼
    with left_col:
        st.header('📝 메모 및 설정')
        
        youtube_url = st.text_input(
            'YouTube URL을 입력하세요:',
            key='youtube_url',
            placeholder='https://www.youtube.com/watch?v=...'
        )

        if st.button('영상 처리 시작', key='process_button'):
            if youtube_url:
                with st.spinner('영상 처리 중...'):
                    try:
                        result = processor.process_video(youtube_url)
                        if 'transcription' in result:
                            st.session_state.current_video = result
                            st.success('영상 처리가 완료되었습니다!')
                        else:
                            st.warning('자막 데이터를 가져오지 못했습니다.')
                    except Exception as e:
                        st.error(f'영상 처리 중 오류가 발생했습니다: {str(e)}')
            else:
                st.warning('YouTube URL을 입력해주세요.')

        # 메모 입력
        st.subheader('✏️ 메모하기')
        note_content = st.text_area('메모를 입력하세요:', height=200, placeholder='여기에 메모를 작성하세요...')
        if st.button('메모 저장'):
            try:
                video_info = st.session_state.current_video.get('video_info', None)
                if note_content:
                    st.session_state.note_manager.save_note(note_content, video_info)
                    st.success('메모가 저장되었습니다!')
                    rain(emoji="✅", font_size=54, falling_speed=5, animation_length="3")
                else:
                    st.warning('메모를 작성해주세요.')
            except Exception as e:
                st.error(f"메모 저장 중 오류가 발생했습니다: {str(e)}")

    # 오른쪽 컬럼
    with right_col:
        if youtube_url:
            st.video(youtube_url)

        # 검색
        if st.session_state.get('current_video'):
            st.subheader('🔍 검색')
            search_query = st.text_input('검색어를 입력하세요:', key='search_query', placeholder='궁금한 내용을 입력하세요.')
            if st.button('검색', key='search_button'):
                search_result = search_content()
                if search_result:
                    st.write(f"**검색 결과:** {search_result['answer']}")

            # 자막 표시
            st.subheader('📝 전체 자막')
            transcript = st.session_state.current_video.get('transcription', None)
            if transcript:
                for segment in transcript.get('segments', []):
                    st.markdown(f"[{format_time(segment['start'])}] {segment['text']}")

if __name__ == "__main__":
    main()