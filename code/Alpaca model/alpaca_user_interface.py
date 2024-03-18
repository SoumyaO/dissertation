from alpaca_llm import (
    alpaca_chains,
    alpaca_formatter,
    alpaca_predictor,
)

from langchain.schema import AIMessage, HumanMessage

import streamlit as st


def set_chosen_response(input_state, chosen_response):
    st.session_state.state = input_state
    st.session_state.chosen_response = chosen_response


def initialize():
    st.title("Your personal assistant")

    if "state" not in st.session_state:
        st.session_state.state = "listening"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "model_alpaca" not in st.session_state:
        st.session_state.model_alpaca = alpaca_chains()


def print_history(col):
    with col:
        # Print conversation history
        st.subheader("Conversation History")
        for message in st.session_state.chat_history:
            st.write(message)


def main():
    initialize()

    # Split into two columns
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        # Receive message
        st.session_state.received_message = st.text_input(
            "Type the message here and click get suggestions button"
        )

        if st.button("Get suggestions") and st.session_state.received_message != "":
            st.divider()
            responses = alpaca_predictor(
                st.session_state.model_alpaca,
                st.session_state.received_message,
                st.session_state.chat_history,
            )
            formatted_response = alpaca_formatter(
                responses[0], responses[1], responses[2]
            )

            # Suggestion buttons
            buttons = {
                tone: st.button(
                    f"{tone}: {formatted_response[tone]}",
                    on_click=set_chosen_response,
                    args=("response_chosen", formatted_response[tone]),
                )
                for tone in ("Positive", "Neutral", "Negative")
            }

        if st.session_state.state == "response_chosen":
            st.write(f"Your response: {st.session_state.chosen_response}")

            # Update history with the chosen message
            st.session_state.chat_history.append(
                "\nThem: " + st.session_state.received_message
            )
            st.session_state.chat_history.append(
                "\nYou: " + st.session_state.chosen_response
            )

            # Update memory of the models
            st.session_state.model_alpaca["positive_chain"].memory.chat_memory.messages[
                -1
            ] = AIMessage(content=st.session_state.chosen_response)
            st.session_state.model_alpaca["neutral_chain"].memory.chat_memory.messages[
                -1
            ] = AIMessage(content=st.session_state.chosen_response)
            st.session_state.model_alpaca["negative_chain"].memory.chat_memory.messages[
                -1
            ] = AIMessage(content=st.session_state.chosen_response)
            st.session_state.state = "listening"
        if st.button("Clear History"):
            st.session_state.model_alpaca["positive_chain"].memory.clear()
            st.session_state.model_alpaca["neutral_chain"].memory.clear()
            st.session_state.model_alpaca["negative_chain"].memory.clear()

            st.session_state.state = "listening"
            st.write("Conversation history cleared!")

    print_history(col2)


if __name__ == "__main__":
    main()
