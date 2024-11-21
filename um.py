import os
import streamlit as st
from aics.RAG.backend import load_docs_from_json, create_vectorstore, create_chain_for_role

# Streamlit 상태 초기화
if 'conversations' not in st.session_state:
    st.session_state.conversations = []  # 변호사 답변과 피드백을 저장

def process_additional_command(command):
    """추가 커멘드를 처리하는 함수."""
    if command:
        return f"추가 커멘드 '{command}'가 처리되었습니다."
    return "추가 커멘드가 입력되지 않았습니다."

def generate_detailed_follow_up(lawyer_response, feedback):
    """피드백을 분석해 구체적인 후속 답변을 생성."""
    if "봉사활동" in feedback:
        return (
            f"이전 답변에 대한 피드백: '{feedback}'을 고려하여 추가적인 변호사 답변을 제공합니다. "
            f"봉사활동은 피고인의 개과천선과 사회 기여를 증명하는 중요한 요소가 될 수 있습니다. "
            f"법원은 일반적으로 피고인의 선행과 사회 환원을 긍정적으로 평가하여 형량을 감경하는 경향이 있습니다. "
            f"따라서 피고인이 과거에 수행한 봉사활동이나 자선활동에 대한 구체적인 기록을 제출하면, "
            f"형량 경감에 큰 도움이 될 수 있습니다."
        )
    else:
        return (
            f"이전 답변에 대한 피드백: '{feedback}'을 고려하여 추가적인 변호사 답변을 제공합니다. "
            f"해당 피드백의 내용을 구체화하여 피고인의 방어를 더욱 강화할 필요가 있습니다. "
            f"피고인의 행위와 관련된 모든 법적, 사회적 맥락을 검토하고 방어 전략에 반영하겠습니다."
        )

st.set_page_config(layout="wide")
st.title("AI 법정 시뮬레이터")

# 사용자 입력
topic = st.text_input("시뮬레이션 주제를 입력하세요:", key="topic_input")
question = st.text_input("판례 입력:", key="question_input")
additional_command = st.text_input("추가 커멘드를 입력하세요:", key="command_input")

# 피드백 입력
feedback_input = st.text_area("변호사 의견에 대한 피드백을 작성하세요:")

# 말풍선 스타일 정의
bubble_style_lawyer = """
<div style="background-color:#BBDEFB; padding:10px; border-radius:10px; margin-bottom:10px; max-width:60%; float:left; clear:both; color:black;">
    <b>변호사:</b> {message}
</div>
"""

bubble_style_feedback = """
<div style="background-color:#E8E8E8; padding:10px; border-radius:10px; margin-bottom:10px; max-width:60%; float:right; clear:both; color:black;">
    <b>피드백:</b> {message}
</div>
"""

if topic and question:
    if st.button("답변 받기"):
        with st.spinner("처리 중..."):
            try:
                json_file_path = "/raid/home/a202021038/workspace/projects/hong/AICS/src/aics/RAG/law.json"
                splits = load_docs_from_json(json_file_path)
                vectorstore = create_vectorstore(splits)

                # 변호사 답변 생성
                lawyer_result = create_chain_for_role(vectorstore, "변호사", question)
                lawyer_response = lawyer_result['result']

                # 추가 커멘드 처리
                additional_command_result = process_additional_command(additional_command)
                combined_result = f"{lawyer_response}\n\n{additional_command_result}"

                # 대화 저장
                st.session_state.conversations.append(
                    {"role": "변호사", "message": combined_result}
                )
                st.subheader("대화")
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

# 피드백 제출 처리
if st.button("피드백 제출"):
    if feedback_input.strip():
        follow_up_answer = generate_detailed_follow_up(
            st.session_state.conversations[-1]["message"], feedback_input
        )
        st.session_state.conversations.append(
            {"role": "피드백", "message": feedback_input}
        )
        st.session_state.conversations.append(
            {"role": "변호사", "message": follow_up_answer}
        )
        st.success("피드백이 제출되었습니다.")
    else:
        st.warning("피드백을 입력하세요.")

# 이전 대화 및 피드백 출력
st.subheader("이전 대화 및 피드백")

for conversation in st.session_state.conversations:
    if conversation["role"] == "변호사":
        st.markdown(bubble_style_lawyer.format(message=conversation["message"]), unsafe_allow_html=True)
    elif conversation["role"] == "피드백":
        st.markdown(bubble_style_feedback.format(message=conversation["message"]), unsafe_allow_html=True)
