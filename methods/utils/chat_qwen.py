from methods.utils import OpenAI


def chat_with_reasoning(system_prompt, user_prompt, model='Qwen/QwQ-32B'):
    client = OpenAI(
        base_url='https://api-inference.modelscope.cn/v1/',
        api_key='YOUR_API_KEY',  # ModelScope Token
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        temperature=0.6,
        top_p=0.95,
        max_completion_tokens=100,
        stream=True,
    )
    for i in range(10):
        try:
            done_reasoning = False
            reasoning = ""
            answer = ""
            for chunk in response:
                reasoning_chunk = chunk.choices[0].delta.reasoning_content
                answer_chunk = chunk.choices[0].delta.content
                if reasoning_chunk != '':
                    reasoning += reasoning_chunk
                    print(reasoning_chunk, end='', flush=True)
                elif answer_chunk != '':
                    if not done_reasoning:
                        print('\n\n === Final Answer ===\n')
                        done_reasoning = True
                    print(answer_chunk, end='', flush=True)
                    answer += answer_chunk
            return reasoning, answer
        except Exception as e:
            print(f"与{model}对话时发生错误: {str(e)}...第{i+1}次重试")
    return None, None


def chat(system_prompt, user_prompt, model="qwen-turbo"):
    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="YOUR_API_KEY",
    )
    for i in range(10):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                modalities=["text"],
                temperature=0.6,
                top_p=0.95,
                max_completion_tokens=30,
                stream_options={"include_usage": True},
                seed=42,
                stream=True,
            )
            text = ""
            for chunk in completion:
                if chunk.choices:
                    if chunk.choices[0].delta.content is not None:
                        text += chunk.choices[0].delta.content
                else:
                    num_tokens = chunk.usage
                    print(chunk.usage.completion_tokens)
            return text
        except Exception as e:
            print(f"与{model}对话时发生错误: {str(e)}...第{i+1}次重试")
    return None


if __name__ == "__main__":
    response = client.chat.completions.create(
        model='Qwen/QwQ-32B',  # ModelScope Model-Id
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '你好'},
        ],
        stream=True,
    )
    done_reasoning = False
    for chunk in response:
        reasoning_chunk = chunk.choices[0].delta.reasoning_content
        answer_chunk = chunk.choices[0].delta.content
        if reasoning_chunk != '':
            print(reasoning_chunk, end='', flush=True)
        elif answer_chunk != '':
            if not done_reasoning:
                print('\n\n === Final Answer ===\n')
                done_reasoning = True
            print(answer_chunk, end='', flush=True)
