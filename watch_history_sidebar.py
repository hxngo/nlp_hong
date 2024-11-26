import streamlit as st
import pandas as pd
from datetime import datetime

def show_watch_history_sidebar():
    """시청 기록을 사이드바에 표시하는 함수"""
    st.sidebar.header('📺 시청 기록')
    
    if 'processor' in st.session_state:
        history = st.session_state.processor.content_analyzer.user_history
        if history:
            # 데이터프레임 생성
            history_df = pd.DataFrame([
                {
                    'Watched At': item['timestamp'],
                    '제목': item['title'],
                    '영상 ID': item['video_id']
                } for item in history
            ])
            
            # 시청 시간 기준으로 정렬 (최신순)
            history_df = history_df.sort_values('Watched At', ascending=False)
            
            # 데이터프레임 표시
            st.sidebar.dataframe(
                history_df,
                hide_index=True,
                column_config={
                    'Watched At': st.column_config.DatetimeColumn(
                        'Watched At',
                        format='YYYY-MM-DD HH:mm',
                        width='medium'
                    ),
                    '제목': st.column_config.Column(
                        width='large'
                    )
                },
                height=300  # 높이 제한
            )
            
            # 전체 기록 삭제 버튼
            if st.sidebar.button('전체 기록 삭제', key='clear_history'):
                st.session_state.processor.content_analyzer.user_history = []
                st.sidebar.success('시청 기록이 삭제되었습니다.')
                st.rerun()
        else:
            st.sidebar.info('아직 시청 기록이 없습니다.')
    else:
        st.sidebar.warning('시스템이 초기화되지 않았습니다.')