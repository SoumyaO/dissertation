import time
import streamlit as st
from langchain.schema import AIMessage, HumanMessage

from llm_ft import ChatOutputFormatter, get_conversation_chain


def divider():
    st.write('------------------')


def print_history():
    with col2:
        # Print conversation history
        st.subheader('Conversation History')
        for message in st.session_state.conversation.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                st.write(f'Them: {message.content}')
            else:
                st.write(f'You: {message.content}')


def set_chosen_response(stage, chosen_response):
    st.session_state.stage = stage
    st.session_state.chosen_response = chosen_response


# Set up stage (how the user is progressing through the app)
if 'stage' not in st.session_state:
    st.session_state.stage = 'listening'
    st.avg_time = 0
    st.msg_count = 0

# Initialise conversation and persist it across sessions
if 'conversation' not in st.session_state:
    st.session_state.conversation = get_conversation_chain(summary=False)

output_formatter = ChatOutputFormatter()

# ------------- APP ------------- #
st.markdown(
    "<h1 style='text-align: center; '>Demo app</h1>",
    unsafe_allow_html=True,
)

# Split into two columns
col1, col2 = st.columns(2, gap='medium')

with col1:
    # Receive message
    received_message = st.text_input('Input people\'s message here')

    # let the user say with just a few words and have chat gpt complete the rest
    # Or let the user ask chat gpt for an appropriate response, or type a response
    # themselves

    # When the user click on 'Get suggestions'
    if st.button('Get suggestions'):
        divider()

        if received_message:
            start_time = time.time()
            response = st.session_state.conversation.predict(
                input=received_message)
            time_elapsed = time.time() - start_time
            st.msg_count += 1
            st.avg_time = (st.avg_time * (st.msg_count - 1) + time_elapsed) / st.msg_count
            st.write(f'Time taken: {time_elapsed}s, average time: {st.avg_time}s')
            formatted_response = output_formatter.format_message(response)

            # Show 3 suggestions
            buttons = {
                tone: st.button(
                    f'{tone}: {formatted_response[tone]}',
                    on_click=set_chosen_response,
                    args=('response_chosen', formatted_response[tone]),
                )
                for tone in ('Positive', 'Neutral', 'Negative')
            }

            st.session_state.stage = 'choose_response'

        else:
            st.write('Please enter a message and click on \'Get suggestions\'')

    if st.session_state.stage == 'response_chosen':
        chosen_response = st.session_state.chosen_response

        st.session_state.conversation.memory.chat_memory.messages[-1] = AIMessage(
            content=chosen_response,
        )

        st.write(f'Your response: {chosen_response}')

    divider()
    # Clear conversation history button
    if st.button('Clear History'):
        st.session_state.conversation.memory.clear()
        st.session_state.stage = 'listening'
        st.write('Conversation history cleared!')

    # # Text completion button
    # if st.button('Text Completion'):
    #     if user_input:
    #         completion = complete_text(user_input)
    #         st.write(f'Completion: {completion}')
    #         user_input = ''

print_history()
