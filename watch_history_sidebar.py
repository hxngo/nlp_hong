import streamlit as st
import pandas as pd

def show_watch_history_sidebar():
    """사이드바에 시청 기록 섹션을 표시합니다."""
    st.sidebar.header('📚 시청 기록')
    
    if st.sidebar.button('시청 기록 보기', key='view_history_sidebar'):
        if 'processor' in st.session_state:
            history = st.session_state.processor.content_analyzer.user_history
            if history:
                history_df = pd.DataFrame([
                    {
                        '시청 시간': item['timestamp'],
                        '제목': item['title'],
                        '영상 ID': item['video_id']
                    } for item in history
                ])
                st.sidebar.dataframe(
                    history_df,
                    hide_index=True,
                    column_config={
                        '시청 시간': st.column_config.DatetimeColumn(
                            'Watched At',
                            format='YYYY-MM-DD HH:mm'
                        )
                    }
                )
            else:
                st.sidebar.info('아직 시청 기록이 없습니다.')
