from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
import json

####################
## PROMPTS COMBINED
####################


def get_chat_prompt_combined():
    system_template = """ The following is a conversation between two people, a normal person A and a person with motor neuron disease B.
    The user giving the input is A and you are B. You should respond to A in a friendly manner.
    Keep in mind that B cannot do any physically intense activity.
    For every input you must provide one Positive, one Neutral and one Negative response formatted as a json object, with 3 keys: Positive, Neutral, Negative respectively.
    The value of each key should be a string, which is the response in that tone. The response must not be left empty.

    Current conversation:
    {history}
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    return chat_prompt


def get_chat_prompt_combined_2():
    system_template = """The following is a friendly conversation between a person (A) and another person with Motor neuron disease (B). 
    B is talkative and provides lots of specific details from its context. 
    If B does not know the answer to a question, they truthfully say they do not know.
    Your task is to suggest B 3 answers in 3 different tone of voices: Positive, Neutral, Negative. 
    Your suggestions need to be in a json object, with 3 keys: Positive, Neutral, Negative. 
    The value of each key should be a string, which is the response in that tone. All 3 keys must always be present.
 """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = """ Conversation:
    {history}
    A: {input}
    B: """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    return chat_prompt


###################
## PROMPTS BY TONE
###################


def get_chat_prompt_by_tone_orca(tone):
    system_template = (
        """ The following is a conversation between two people, a normal person A and a person with motor neuron disease B.
    The user giving the input is A and you are B. You should respond to A in a friendly manner.
    Keep in mind that B cannot do any physically intense activity.
    You must provide a response in a """
        + tone
        + """ tone. The response must not be left empty. Make sure not to mention about being a patient with MND.


    Current conversation:
    {history}
    """
    )

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    return chat_prompt


def get_chat_prompt_by_tone_alpaca(tone):
    system_template = (
        """ You follow instructions very well. The user inputs a conversation between two people, a normal person A and a person with motor neuron disease B. 
        Respond as if you are B in a single sentence only. The tone should always be """
        + tone
        + """. There should always be a response. B cannot do intense physical activity but do not mention this explicitly in the response."""
    )

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = """ Conversation:
    {history}
    A: {input}
    B: """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    return chat_prompt


def get_chat_prompt_by_tone(tone):
    system_template = (
        """ The following is a conversation between two people, a normal person A and a person with motor neuron disease B.
    For the input given by A respond as B in a """
        + tone
        + """ tone.
    Keep in mind that B cannot do any physically intense activity.

    Current conversation:
    {history}
    """
    )

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    return chat_prompt


def get_chat_prompt_by_tone_2(tone):
    system_template = (
        """ Read the conversation below and reply to the last message as if you are a patient with motor neuron disease in a """
        + tone
        + """ tone.
    Keep in mind that you cannot do any physically intense activity. Keep your response to a single sentence. Make sure not to mention about being a patient with MND.

    # Current conversation:
    # {history}
    # """
    )

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    return chat_prompt


def get_chat_prompt_by_tone_3(tone):
    chat_template = (
        """ You follow instructions very well. The user inputs a conversation between two people, a normal person A and a person with motor neuron disease B. 
        Respond as if you are B in a single sentence only. The tone should always be """
        + tone
        + """. There should always be a response. B cannot do intense physical activity but do not mention this explicitly in the response."""
        """ Conversation:
    {history}
    A: {input}
    B: """
    )

    chat_prompt = ChatPromptTemplate.from_template(template=chat_template)

    return chat_prompt


###########
## MEMORY
###########
def get_memory(llm):
    memory = ConversationSummaryBufferMemory(
        max_token_limit=100, llm=llm, ai_prefix="B: ", human_prefix="A: "
    )
    return memory


##################
## Parsers
##################
def parse_response_from_json(response) -> dict:
    """
    Parses the response from a json object to a python dictionary
    @param response: the input response
    @return: the parsed response
    """
    parsed_response = json.loads(response)
    return parsed_response


def parse_response_multiple(
    positive_response, neutral_response, negative_response
) -> dict:
    """
    Parses the response from three separate strings to a python dictionary
    @param positive_response: the positive response
    @param neutral_response: the neutral response
    @param negative_response: the negative response
    @return: the parsed response
    """
    parsed_response = {}
    # if the output is in multiple lines, then only pick the first line
    if "\n" in positive_response:
        positive_response = positive_response.split("\n")[0]
    if "\n" in neutral_response:
        neutral_response = neutral_response.split("\n")[0]
    if "\n" in negative_response:
        negative_response = negative_response.split("\n")[0]

    # if there is a : in the response, then only pick the part after the :
    if ":" in positive_response:
        parsed_response["Positive"] = positive_response.split(": ")[1]
    else:
        parsed_response["Positive"] = positive_response

    if ":" in neutral_response:
        parsed_response["Neutral"] = neutral_response.split(": ")[1]
    else:
        parsed_response["Neutral"] = neutral_response

    if ":" in negative_response:
        parsed_response["Negative"] = negative_response.split(": ")[1]
    else:
        parsed_response["Negative"] = negative_response

    return parsed_response


def parse_response_multiple_2(
    positive_response, neutral_response, negative_response
) -> dict:
    """
    Parses the response from three separate strings to a python dictionary
    @param positive_response: the positive response
    @param neutral_response: the neutral response
    @param negative_response: the negative response
    @return: the parsed response
    """
    parsed_response = {}
    # if the output is in multiple lines, then only pick the first line
    if "\n" in positive_response:
        positive_response = positive_response.split("\n    ")[1]
    if "\n" in neutral_response:
        neutral_response = neutral_response.split("\n    ")[1]
    if "\n" in negative_response:
        negative_response = negative_response.split("\n    ")[1]

    # if there is a : in the response, then only pick the part after the :
    if ":" in positive_response:
        parsed_response["Positive"] = positive_response.split(": ")[1]
    else:
        parsed_response["Positive"] = positive_response

    if ":" in neutral_response:
        parsed_response["Neutral"] = neutral_response.split(": ")[1]
    else:
        parsed_response["Neutral"] = neutral_response

    if ":" in negative_response:
        parsed_response["Negative"] = negative_response.split(": ")[1]
    else:
        parsed_response["Negative"] = negative_response

    return parsed_response