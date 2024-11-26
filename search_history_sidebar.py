import streamlit as st
from datetime import datetime

def show_search_history_sidebar():
    """사이드바에 검색 기록을 표시하는 개선된 컴포넌트"""
    st.sidebar.subheader('📜 검색 기록')
    
    if 'search_history' in st.session_state and st.session_state.search_history:
        for idx, entry in enumerate(reversed(st.session_state.search_history)):
            with st.sidebar.expander(
                f"🔍 검색 #{len(st.session_state.search_history)-idx}", 
                expanded=(idx == 0)  # 최신 검색은 자동으로 펼치기
            ):
                # 시간 정보
                st.markdown(f"**시간**: {entry['timestamp']}")
                
                # 검색어와 답변을 구분된 섹션으로 표시
                st.markdown("---")
                st.markdown("**💭 검색어**")
                st.info(entry['query'])  # info 스타일로 변경
                
                st.markdown("---")
                st.markdown("**🤖 답변**")
                st.success(entry['answer'])  # success 스타일로 변경, 짙은 배경색과 어두운 글자색
                
                # 작업 버튼들
                col1, col2 = st.columns(2)
                with col1:
                    if st.button('🔄 재검색', key=f'reuse_search_{idx}'):
                        st.session_state.search_query = entry['query']
                        st.rerun()
                with col2:
                    if st.button('🗑️ 삭제', key=f'delete_search_{idx}'):
                        st.session_state.search_history.pop(len(st.session_state.search_history)-1 - idx)
                        st.rerun()
        
        # 전체 삭제 버튼
        if st.sidebar.button('🗑️ 전체 기록 삭제', use_container_width=True):
            st.session_state.search_history = []
            st.rerun()
    else:
        st.sidebar.info('검색 기록이 없습니다.')