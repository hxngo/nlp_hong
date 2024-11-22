import streamlit as st

def show_watch_history_sidebar():
    st.subheader("📺 시청 기록")
    if hasattr(st.session_state.processor.content_analyzer, 'user_history'):
        history = st.session_state.processor.content_analyzer.user_history
        if history and len(history) > 0:
            for idx, item in enumerate(history):
                with st.expander(f"🎥 {item['title'][:30]}..."):
                    st.write(f"시청 시간: {item['timestamp']}")
                    video_url = f"https://www.youtube.com/watch?v={item['video_id']}"
                    st.markdown(f"[영상 보기]({video_url})")
                    
                    # 삭제 버튼에 고유한 key 추가 (idx 활용)
                    if st.button('삭제', key=f"delete_history_{item['video_id']}_{idx}"):
                        st.session_state.processor.content_analyzer.remove_from_history(item['video_id'])
                        st.success('시청 기록이 삭제되었습니다!')
                        st.rerun()
        else:
            st.info('시청 기록이 없습니다.')

