import streamlit as st
from datetime import datetime
import pandas as pd
from backend import NoteManager

st.set_page_config(
    page_title="메모 관리",
    page_icon="📝",
    layout="wide"
)

# 스타일 설정
st.markdown("""
    <style>
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
    </style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'note_manager' not in st.session_state:
    st.session_state.note_manager = NoteManager()

def main():
    st.title('📝 메모 관리')
    
    # 필터링 옵션
    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input('🔍 메모 검색', placeholder='검색어를 입력하세요...')
    with col2:
        sort_option = st.selectbox(
            '정렬 기준',
            ['최신순', '오래된순']
        )
    
    # 메모 목록 가져오기
    notes = st.session_state.note_manager.get_notes()
    
    # 검색어로 필터링
    if search_term:
        notes = [
            note for note in notes
            if search_term.lower() in note['content'].lower() or
               search_term.lower() in note['video_info'].get('title', '').lower()
        ]
    
    # 정렬
    if sort_option == '오래된순':
        notes = sorted(notes, key=lambda x: x['created_at'])
    else:  # 최신순
        notes = sorted(notes, key=lambda x: x['created_at'], reverse=True)
    
    # 메모 표시
    if notes:
        for note in notes:
            with st.container():
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"""
                    <div class="memo-card">
                        <div class="memo-metadata">
                            📅 {note['timestamp']}
                            {f"| 📺 {note['video_info'].get('title', '')}" if note['video_info'].get('title') else ""}
                        </div>
                        <div style="margin-top: 0.5rem;">
                            {note['content']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button('삭제', key=f"delete_{note['timestamp']}"):
                        st.session_state.note_manager.remove_note(note['timestamp'])
                        st.rerun()
    else:
        st.info('저장된 메모가 없습니다.')
    
    # 통계
    st.divider()
    st.subheader('📊 메모 통계')
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("총 메모 수", len(notes))
    
    with col2:
        if notes:
            latest_note = max(notes, key=lambda x: x['created_at'])
            st.metric("최근 메모 작성", latest_note['timestamp'])

    # 메모 내보내기
    st.divider()
    if st.button('📥 메모 내보내기 (CSV)'):
        df = pd.DataFrame(notes)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="CSV 파일 다운로드",
            data=csv,
            file_name=f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()