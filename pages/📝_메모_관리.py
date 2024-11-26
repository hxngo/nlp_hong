# /raid/home/a202021038/workspace/projects/hong/AICS/src/aics/RAG/pages/📝_메모_관리.py
import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend_youtube6 import NoteManager

# 페이지 설정
st.set_page_config(
    page_title="메모 관리",
    page_icon="📝",
    layout="wide"
)

# 세션 상태 초기화
if 'note_manager' not in st.session_state:
    st.session_state.note_manager = NoteManager()

# 메인 페이지
st.title('📝 메모 관리')

# 필터링 옵션들
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    search_term = st.text_input('🔍 메모 검색', placeholder='검색어를 입력하세요...')
with col2:
    sort_option = st.selectbox(
        '정렬 기준',
        ['최신순', '오래된순']
    )
with col3:
    # 태그 필터링
    all_tags = ['전체'] + st.session_state.note_manager.get_all_tags()
    selected_tag = st.selectbox('태그 필터', all_tags)

try:
    # 메모 목록 가져오기
    if selected_tag == '전체':
        notes = st.session_state.note_manager.get_notes()
    else:
        notes = st.session_state.note_manager.get_notes(tag=selected_tag)
    
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
        for idx, note in enumerate(notes):
            with st.expander(f"📝 메모 #{len(notes)-idx}", expanded=(idx == 0)):
                # 시간 정보
                st.caption(f"작성 시간: {note['timestamp']}")
                
                # 태그 표시
                if note.get('tags'):
                    tag_cols = st.columns(len(note['tags']))
                    for i, tag in enumerate(note['tags']):
                        tag_cols[i].markdown(f"🏷️ `{tag}`")
                    st.markdown("---")
                
                # 비디오 정보가 있는 경우 표시
                if note['video_info'] and note['video_info'].get('title'):
                    st.caption(f"📺 관련 영상: {note['video_info'].get('title')}")
                
                # 메모 내용
                st.write(note['content'])
                
                # 작업 버튼들
                col1, col2 = st.columns([4, 1])
                with col2:
                    if st.button('🗑️ 삭제', key=f"delete_note_{idx}", use_container_width=True):
                        st.session_state.note_manager.remove_note(note['timestamp'])
                        st.rerun()
    else:
        st.info('저장된 메모가 없습니다.')

    if notes:  # 메모가 있을 때만 통계와 내보내기 옵션 표시
        # 통계
        st.divider()
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            st.metric("총 메모 수", len(notes))
        with stats_col2:
            latest_note = max(notes, key=lambda x: x['created_at'])
            st.metric("최근 메모 작성", latest_note['timestamp'])
        with stats_col3:
            total_tags = len(st.session_state.note_manager.get_all_tags())
            st.metric("사용 중인 태그", f"{total_tags}개")

        # 메모 내보내기
        st.divider()
        if st.button('📥 메모 내보내기 (CSV)', use_container_width=True):
            # DataFrame 생성 시 태그 정보 포함
            export_data = []
            for note in notes:
                note_data = {
                    '작성시간': note['timestamp'],
                    '내용': note['content'],
                    '태그': ', '.join(note.get('tags', [])),
                    '관련 영상': note['video_info'].get('title', '')
                }
                export_data.append(note_data)
            
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="CSV 파일 다운로드",
                data=csv,
                file_name=f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
except Exception as e:
    st.error(f"메모를 불러오는 중 오류가 발생했습니다: {str(e)}")
    st.error("자세한 오류 정보:")
    st.exception(e)