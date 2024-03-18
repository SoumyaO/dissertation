from dotenv import find_dotenv, load_dotenv
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())


def get_memory(summary=False):
    memory_kwargs = {
        "human_prefix": "A",
        "ai_prefix": "B",
    }

    if summary:
        return ConversationSummaryBufferMemory(max_token_limit=100, **memory_kwargs)
    else:
        return ConversationBufferMemory(**memory_kwargs)


def get_conversation_chain(summary=False):
    system_template = """
    The user inputs a friendly conversation between two people, a normal person (A) and a person with motor neuron disease (B). 
    B is talkative and is capable of understanding context and remembering specific details from the conversation. 
    You are playing the role of B and must respond to the query of A with 3 answers in 3 different tones of voices: Positive, Neutral, Negative. 
    Make sure that you are aware of your limitations as a person with motor neuron disease but do not mention it explicitly in your resonses. 
    Your suggestions need to be in a json object, with 3 keys: Positive, Neutral, Negative. 
    The value of each key should be a string, which is the response in that tone. All 3 keys must always be present.
    {history}
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chat_memory = get_memory(summary=summary)
    chat_llm = ChatOpenAI(temperature=0.1, client="gpt-3.5-turbo")
    # chat_llm = ChatOpenAI(temperature=0.1, client="ft:gpt-3.5-turbo-0613:bayes:arp-finetune:7sf3eo7j")

    conversation = ConversationChain(
        prompt=chat_prompt,
        llm=chat_llm,
        memory=chat_memory,
        verbose=True,
    )

    return conversation


class ChatOutputFormatter:
    def __init__(self):
        output_parsing_template = """
            Take the following text and reformat it. Do not change the text itself, only the format.

            text: {text}

            {format_instructions}
        """

        self.output_parser = StructuredOutputParser.from_response_schemas(
            [
                ResponseSchema(
                    name="Positive", description="Response in a positive tone"
                ),
                ResponseSchema(
                    name="Neutral", description="Response in a neutral tone"
                ),
                ResponseSchema(
                    name="Negative", description="Response in a negative tone"
                ),
            ]
        )
        self.format_instructions = self.output_parser.get_format_instructions()
        self.output_parser_ai = ChatOpenAI(
            temperature=0.0,
            client="gpt-3.5-turbo",
        )
        self.output_format_prompt = ChatPromptTemplate.from_template(
            template=output_parsing_template,
        )

    def format_message(self, message):
        message_format_prompt = self.output_format_prompt.format_messages(
            text=message,
            format_instructions=self.format_instructions,
        )

        formatted_message = self.output_parser.parse(
            self.output_parser_ai(message_format_prompt).content,
        )

        return formatted_message
