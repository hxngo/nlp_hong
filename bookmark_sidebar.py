import streamlit as st
from datetime import datetime

def show_bookmark_sidebar():
    """사이드바에 북마크 섹션을 표시합니다."""
    st.sidebar.header('📚 북마크')
    
    if 'bookmark_manager' in st.session_state:
        bookmarks = st.session_state.bookmark_manager.get_bookmarks()
        if bookmarks:
            for idx, bookmark in enumerate(bookmarks):
                with st.sidebar.expander(f"📌 북마크 {idx+1} - {bookmark['timestamp']}"):
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
            st.sidebar.info('저장된 북마크가 없습니다.')
