# watch_history_sidebar.py
import streamlit as st

def show_watch_history_sidebar():
   """시청 기록을 사이드바에 표시하는 함수"""
   st.sidebar.header('📚 시청 기록')
   
   if 'processor' in st.session_state:
       history = st.session_state.processor.content_analyzer.user_history
       if history:
           # 최신순으로 정렬
           for idx, item in enumerate(reversed(history)):
               # 각 기록을 expander로 표시
               with st.sidebar.expander(f"📺 {item['title']}", expanded=False):
                   st.write(f"시청 시간: {item['timestamp']}")
                   video_url = f"https://www.youtube.com/watch?v={item['video_id']}"
                   st.markdown(f"[영상으로 이동]({video_url})")
           
           # 전체 기록 삭제 버튼
           if st.sidebar.button('전체 기록 삭제', use_container_width=True, key='clear_all_history'):
               st.session_state.processor.content_analyzer.user_history = []
               st.sidebar.success('시청 기록이 삭제되었습니다.')
               st.rerun()
       else:
           st.sidebar.info('아직 시청 기록이 없습니다.')