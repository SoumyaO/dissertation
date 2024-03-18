from helpers import (
    get_chat_prompt_by_tone_alpaca,  # chat prompt
    get_memory,  # memory
    parse_response_multiple,  # parser
)

from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
    pipeline,
    BitsAndBytesConfig,
)

import time


def get_alpaca_llm():
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

    tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native", legacy=False)
    alpaca_llm = LlamaForCausalLM.from_pretrained(
        "chavinlo/alpaca-native",
        load_in_8bit=True,
        device_map="cpu",
        quantization_config=quantization_config,
    )

    generation_config = GenerationConfig(
        temperature=0.6,
        quantization_config=quantization_config,
        load_in_8bit=True,
        top_p=0.95,
        repetition_penalty=1.2,
    )

    pipe = pipeline(
        "text-generation",
        model=alpaca_llm,
        tokenizer=tokenizer,
        max_length=512,
        generation_config=generation_config,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


def alpaca_chains():
    llm = get_alpaca_llm()

    positive_chain = LLMChain(
        prompt=get_chat_prompt_by_tone_alpaca("positive"),
        llm=llm,
        memory=get_memory(llm),
        verbose=True,
    )

    neutral_chain = LLMChain(
        prompt=get_chat_prompt_by_tone_alpaca("neutral"),
        llm=llm,
        memory=get_memory(llm),
        verbose=True,
    )

    negative_chain = LLMChain(
        prompt=get_chat_prompt_by_tone_alpaca("negative"),
        llm=llm,
        memory=get_memory(llm),
        verbose=True,
    )

    chains = {
        "positive_chain": positive_chain,
        "neutral_chain": neutral_chain,
        "negative_chain": negative_chain,
    }
    return chains


def alpaca_predictor(alpaca_chains, text, chat_history):
    print("text ", text)
    print("chat_history ", chat_history)
    start_time = time.time()
    positive_response = alpaca_chains["positive_chain"].predict(input=text)
    t1 = time.time() - start_time
    print(positive_response)
    start_time = time.time()
    neutral_response = alpaca_chains["neutral_chain"].predict(input=text)
    t2 = time.time() - start_time
    print(neutral_response)
    start_time = time.time()
    negative_response = alpaca_chains["negative_chain"].predict(input=text)
    t3 = time.time() - start_time
    print(negative_response)

    print("avg time: ", (t1 + t2 + t3) / 3)
    return positive_response, neutral_response, negative_response


def alpaca_formatter(positive_response, neutral_response, negative_response):
    return parse_response_multiple(
        positive_response, neutral_response, negative_response
    )
