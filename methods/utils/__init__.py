from openai import OpenAI

from .chat_qwen import chat as chat_stream
from .chat_qwen import chat_with_reasoning as chat_qwq
from .env_utils import load_env
from .history import History
from .logger import setup_logger
from .reward_func import reward_func


def chat(system_prompt, user_prompt, model="deepseek"):
    """
    Function to chat with LLMs.

    Args:
        prompt: Input prompt text

    Returns:
        Response content.
    """

    if model == "openai":
        client = OpenAI(
            api_key="YOUR_API_KEY"
        )
        model = "gpt-4o"
    elif model == "deepseek":
        client = OpenAI(
            api_key="YOUR_API_KEY",
            base_url="https://api.siliconflow.cn/v1",
        )
        model_path = "deepseek-ai/DeepSeek-V3"
    elif model == "kimi":
        client = OpenAI(
            api_key="YOUR_API_KEY",
            base_url="https://api.moonshot.cn/v1",
        )
        model_path = "moonshot-v1-auto"
    elif model == "qwq":
        reasoning, answer = chat_qwq(system_prompt, user_prompt)
        logger.info(f"Reasoning: {reasoning}")
        logger.info(f"Answer: {answer}")
        return answer
    elif model == "deepseek-mscope":
        client = OpenAI(
            api_key="YOUR_API_KEY",
            base_url="https://api-inference.modelscope.cn/v1/",
        )
        model_path = "deepseek-ai/DeepSeek-V3"
    elif model == "deepseek-ali":
        client = OpenAI(
            api_key="YOUR_API_KEY",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        model_path = "deepseek-v3"
    elif model == "qwen-max":
        answer = chat_stream(system_prompt, user_prompt, model="qwen-max-2025-01-25")
        return answer
    elif model == "qwen-omni-turbo":
        answer = chat_stream(system_prompt, user_prompt)
        return answer
    else:
        raise ValueError("Unknown model: {}".format(model))

    for i in range(10):
        try:
            response = client.chat.completions.create(
                model=model_path,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.6,
                top_p=0.95,
                seed=42,
                max_completion_tokens=30,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"与{model}对话时发生错误: {str(e)}...第{i+1}次重试")

    return None
