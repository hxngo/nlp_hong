import streamlit as st
import pandas as pd

def show_search_history_sidebar():
    """사이드바에 검색 기록 섹션을 표시합니다."""
    st.sidebar.header('📜 검색 기록')
    
    if 'search_history' in st.session_state and st.session_state.search_history:
        history_df = pd.DataFrame(st.session_state.search_history)
        st.sidebar.dataframe(
            history_df,
            column_config={
                'timestamp': '시간',
                'query': '검색어',
                'answer': '답변'
            },
            hide_index=True
        )
    else:
        st.sidebar.info('검색 기록이 없습니다.')
