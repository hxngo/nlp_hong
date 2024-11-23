import streamlit as st
import os
from datetime import datetime
import pandas as pd
from backend_youtube6 import VideoProcessor, BookmarkManager, NoteManager, TranscriptManager, YouTubeExtractor
from bookmark_sidebar import show_bookmark_sidebar
from search_history_sidebar import show_search_history_sidebar
from watch_history_sidebar import show_watch_history_sidebar
from streamlit_extras.let_it_rain import rain
from show_watch_history_sidebar import show_watch_history_sidebar

# 환경 변수 설정
if 'OPENAI_API_KEY' not in os.environ:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

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
def initialize_session_state():
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
    if 'processor' not in st.session_state:
        try:
            # API 키 확인
            youtube_api_key = os.getenv('YOUTUBE_API_KEY')
            if not youtube_api_key:
                st.error('YouTube API 키가 설정되지 않았습니다. 환경 변수를 확인해주세요.')
                st.stop()
            
            # VideoProcessor 초기화
            st.session_state.processor = VideoProcessor()
            
        except Exception as e:
            st.error(f'VideoProcessor 초기화 실패: {str(e)}')
            st.stop()

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

def main():
    # 세션 상태 초기화 및 VideoProcessor 인스턴스 생성
    initialize_session_state()
    processor = st.session_state.processor

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
            if youtube_url:
                with st.spinner('영상 처리 중...'):
                    try:
                        result = processor.process_video(youtube_url)
                        if 'transcription' in result and result['transcription']:
                            st.session_state.current_video = result
                            st.success('영상 처리가 완료되었습니다!')
                        else:
                            st.warning('자막 데이터를 가져오지 못했습니다.')
                    except Exception as e:
                        st.error(f'영상 처리 중 오류가 발생했습니다: {str(e)}')
            else:
                st.warning('YouTube URL을 입력해주세요.')
        
        st.markdown("---")
        
        # 메모 입력 섹션
        st.subheader('✏️ 메모하기')
        note_content = st.text_area(
            '메모를 입력하세요:',
            height=200,
            placeholder='여기에 메모를 작성하세요...',
            key='note_content'
        )
        
        if st.button('메모 저장', key='save_note_button', use_container_width=True):
            video_info = st.session_state.current_video['video_info'] if st.session_state.current_video else None
            if st.session_state.note_manager.save_note(note_content, video_info):
                st.success('메모가 저장되었습니다!')
                rain(
                    emoji="✅",
                    font_size=54,
                    falling_speed=5,
                    animation_length="3"
                )
            else:
                st.warning('메모를 작성해 주세요!')
        
        # 검색 섹션
        st.markdown("---")
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
            
                # 자동 요약 섹션 (한 번만 표시)
                if 'summary' in st.session_state.current_video:
                    st.subheader('📝 영상 요약')
                    summary = st.session_state.current_video['summary']
                
                    # 요약 내용 표시
                    st.write("**요약 내용:**")
                    st.write(summary['summary'])
                
                    # 요약 통계 표시
                    st.markdown(f"""
                    <div style='font-size: 0.8em; color: #666;'>
                    원본 길이: {summary['original_length']} 단어 → 요약 길이: {summary['summary_length']} 단어
                    </div>
                    """, unsafe_allow_html=True)
                
                    # 요약 내용 복사 버튼
                    if st.button('요약 내용 복사', key=f'copy_summary_{youtube_url}'):
                        st.write('요약 내용이 클립보드에 복사되었습니다!')
                        st.code(summary['summary'])

                    # 사이드바에 각 섹션 추가
                    with st.sidebar:
                        st.divider()
                        show_bookmark_sidebar()
        
                        st.divider()
                        show_search_history_sidebar()
        
                        st.divider()
                        show_watch_history_sidebar()

                # 추천 컨텐츠 섹션 
                st.subheader('🎯 추천 컨텐츠')
                if 'recommendations' in st.session_state.current_video:
                    recommendations = st.session_state.current_video['recommendations']
                    if recommendations:
                        for idx, rec in enumerate(recommendations):
                            with st.expander(f"추천 {idx+1}: {rec['title']}"):
                                st.markdown(f"""
                                    **비디오 정보:**
                                    - 제목: {rec['title']}
                                    - 설명: {rec.get('description', 'N/A')}
                                    - 조회수: {rec.get('view_count', 'N/A')}
                                """)
                                # 추천 영상으로 이동 버튼
                                video_url = f"https://www.youtube.com/watch?v={rec['video_id']}"
                                st.markdown(f"[이 영상 보기]({video_url})")
                    else:
                        st.info('추천할 컨텐츠를 찾을 수 없습니다.')
                
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
                    try:
                        transcript = st.session_state.transcript_manager.get_transcript(video_id)
                    
                        if transcript:
                            for segment in transcript:
                                st.markdown(f"""
                                <div class="transcript-segment">
                                    <span class="timestamp">[{format_time(segment['start'])}]</span> {segment['text']}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info('자막 정보가 없습니다. 데이터를 확인해주세요.')
                    except Exception as e:
                        st.error(f'자막 데이터를 가져오는 중 오류가 발생했습니다: {str(e)}')
        
        # 검색 결과
        if 'search_result' in st.session_state and st.session_state.search_result:
            st.subheader('🔍 검색 결과')
            result = st.session_state.search_result
            
            st.markdown(f"**답변:**\n{result['answer']}")

if __name__ == "__main__":
    main()
