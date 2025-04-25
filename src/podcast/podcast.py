import json
import os

import pandas as pd
from typing import List, Dict, Any

import requests
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains import SequentialChain, SimpleSequentialChain
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser
from langchain_together import ChatTogether
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
import pandas as pd

# Configure keys in environment variables
load_dotenv()
llm = ChatTogether(model_name="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo")


# sdk = YCloudML(folder_id=os.getenv("FOLDER_ID"),auth=os.getenv("API_KEY"))
# sdk.setup_default_logging()
# llm = sdk.models.completions('yandexgpt-32k').langchain(model_type="chat", timeout=30)
# llm = ChatTogether(model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
def web_search(query: str):
    load_dotenv()
    from langchain_tavily import TavilySearch

    """Finds general knowledge information using Tavily search."""
    if str(query) == "None":
        return "No information gathered by web_search"
    tool_instance = TavilySearchResults(max_results=5, include_answer=True, include_images=False)
    response = tool_instance.invoke(query)

    return response



# 1. ArticleIngestionAgent: load DataFrame and attach metadata
def ingest_articles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Reads a DataFrame of articles and returns a list of dicts with id, text, and metadata.
    Metadata includes date, author, and section, ensuring historical authenticity.
    """
    articles = []
    for _, row in df.iterrows():
        articles.append({
            "id": row.get("id"),
            "text": row.get("text"),
            "metadata": {
                "date": row.get("date"),
                "author": row.get("author"),
                "section": row.get("section")
            }
        })
    return articles


ingest_tool = Tool(
    name="ArticleIngestionAgent",
    func=ingest_articles,
    description="Ingest articles DataFrame and output id, text, metadata"
)

# 2. TopicSelectionAgent: select relevant articles by topic or date
topic_prompt = PromptTemplate(
    input_variables=["articles", "topic", "date_range"],
    template="""
Приведен следующий список статей с метаданными и текстом:
{articles}

Выбери те, которые наиболее актуальны для эпизода по теме "{topic}" или в пределах "{date_range}".
Убедись, что выбранные материалы сохраняют историческую достоверность и фактическую точность.
Возращай ответ в формате json:
[
{{
    "reasoning": "Краткое изложение вашего мыслительного процесса и факторов, влияющих на ваше решение, должно быть написано с вашей точки зрения. Это сообщение будет показано пользователям, чтобы показать, как вы думаете и принимаете решения"
    "array":  массив идентификаторов статей в формате JSON
}}
]  
"""
)
topic_chain = LLMChain(llm=llm, prompt=topic_prompt)

# 3. SummarizationAgent: generate short and long summaries
summary_prompt = PromptTemplate(
    input_variables=["article_text", "level"],
    template="""
    Сформируйте краткое изложение следующего текста на {level}, сохраняя историческую достоверность и фактологическую точность:

    {article_text}

    Верните краткое изложение в виде обычного текста.
"""
)


def summarize_article(article: Dict[str, Any], levels: List[str] = ["short", "long"]) -> Dict[str, str]:
    summaries = {}
    for level in levels:
        chain = LLMChain(llm=llm, prompt=summary_prompt)
        summary = chain.run(article_text=article["text"], level=level)
        summaries[level] = summary.strip()

    return {article["id"]: summaries}


# 4. SegmentStructureAgent: plan episode segments
segment_prompt = PromptTemplate(
    input_variables=["summaries"],
    template="""
Учитывая краткое содержание статьи:
{summaries}

Создайте подробный план эпизода с сегментами: вступление(должно содержать список ключевых личностей упоминаемых в статьях, кем статьи написаны)
, основная часть, цитаты и заключение.
Возращай ответ в формате json:
[
{{
    "reasoning": "Краткое изложение вашего мыслительного процесса и факторов, влияющих на ваше решение, должно быть написано с вашей точки зрения. Это сообщение будет показано пользователям, чтобы показать, как вы думаете и принимаете решения"
    "segments":  выводи в формате JSON с названиями сегментов и описаниями
}}
] 
"""
)
segment_chain = LLMChain(llm=llm, prompt=segment_prompt)

# 5. QuestionGeneratorAgent: formulate deep-dive questions
question_prompt = PromptTemplate(
    input_variables=["episode_plan", "summaries"],
    template="""
Основываясь на плане эпизода:
{episode_plan}
и кратком изложении статьи:
{summaries}

Сформулируйте 2-3 содержательных вопроса для более глубокого изучения темы.
Возращай ответ в формате json:
[
{{
    "reasoning": "Краткое изложение вашего мыслительного процесса и факторов, влияющих на ваше решение, должно быть написано с вашей точки зрения. Это сообщение будет показано пользователям, чтобы показать, как вы думаете и принимаете решения"
    "questions":  массив вопросов в формате JSON
}}
]
"""
)
question_chain = LLMChain(llm=llm, prompt=question_prompt)

# 6. WebSearchAgent: gather external info
web_search_tool = Tool(
    name="WebSearchAgent",
    func=lambda query: web_search(query),
    description="Search the web for relevant information not found in the articles"
)

# 7. ScriptWritingAgent: writes full script
script_prompt = PromptTemplate(
    input_variables=["episode_plan", "summaries", "questions"],
    template="""
Напишите полный сценарий аудио-подкаста, основаного на освещение событий одного дня на основе старых газет на 5 минут, включающий:

- Введение(озвучь ключевые личности/сущности упоминаемые в газетах)
- Переходы
- Цитаты
- Ответы на эти вопросы: {questions}

Используйте этот план:
{episode_plan}
и краткое содержание:
{summaries}


Обеспечьте историческую достоверность и фактологическую точность.
Возращай ответ в формате json:
[
{{
    "reasoning": "Краткое изложение вашего мыслительного процесса и факторов, влияющих на ваше решение, должно быть написано с вашей точки зрения. Это сообщение будет показано пользователям, чтобы показать, как вы думаете и принимаете решения"
    "script":  текст сценария
}}
]
"""
)
script_chain = LLMChain(llm=llm, prompt=script_prompt)

# 8. QAReviewAgent: fact-check and style review
qa_prompt = PromptTemplate(
    input_variables=["script_text"],
    template="""
Просмотрите следующий сценарий на предмет соответствия фактам и исторической достоверности:

{script_text}

Если таковых нет, ответьте "OK". Если найдено несоответствие по стилю, то тип ошибки style, 
в поле description: советы и предложение по стилю. Если найдено несоответсвие фактаи или исторической достоверности, то тип ошибки fact.
В таком случае для каждой ошибки будет произведен поиск в интернете. В поле description: вопрос для запроса в интернет, чтобы проверить несоответсвие. 
Учти, что вопрос должен не обширным по смыслу и небольшим по размеру, он должен быть направлен на получения ответа на один вопрос.

Возращай ответ в формате json:
[
{{
    "reasoning": "Краткое изложение вашего мыслительного процесса и факторов, влияющих на ваше решение, должно быть написано с вашей точки зрения. Это сообщение будет показано пользователям, чтобы показать, как вы думаете и принимаете решения"
    "issues":  Верните в формате JSON с ключами "issues" (массив) или "status".
}}
]
"""
)
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

editor_prompt = PromptTemplate(
    input_variables=["script_text", "external_info"],
    template="""
Ты агент-правщик сценария. 
Тебе дан текст сценария для аудио-подкаста: {script_text}
Убери из него названия сегментов(вступление, заключение и тому подобное).

Qa агент делал запросы в интернет для проверки нескольких фактов из статей.
Если факты из интернета и факты из статьи различаются, то тебе нужно об этом сказать по такому механизму:
в сценарии пишешь о событии, как сказано в статье. После этого добавляешь, что по результатом поиска в интернете
информация различается и озвучиваешь версию из интернета.

Ниже приведены правки qa агента.
{external_info}

Обеспечь историческую достоверность и фактологическую точность.
Возращай ответ в формате json:
[
{{
    "reasoning": "Краткое изложение вашего мыслительного процесса и факторов, влияющих на ваше решение, должно быть написано с вашей точки зрения. Это сообщение будет показано пользователям, чтобы показать, как вы думаете и принимаете решения"
    "script":  текст сценария с правками
}}
]
"""
)
editor_chain = LLMChain(llm=llm, prompt=editor_prompt)

# 9. SoundDesignAgent: add SSML cues
sound_prompt = PromptTemplate(
    input_variables=["script_text"],
    template="""
Дан сценнарий:
{script_text}



Поддерживаемые теги SSML
На данный момент SpeechKit поддерживает следующие теги SSML:

Описание	Тег
Добавить паузу	<break>
Добавить паузу между параграфами	<p>
Использовать фонетическое произношение	<phoneme>
Корневой тег для текста в формате SSML	<speak>
Добавить паузу между предложениями	<s>
Произношение аббревиатур	<sub>
break
Используйте тег <break>, чтобы добавить в речь паузу заданной длительности. Длительность указывается с помощью атрибутов strength и time. Если эти атрибуты не заданы, то по умолчанию используется strength="medium".

Атрибут	Описание
strength	Длительность паузы, зависит от контекста. Допустимые значения:
* weak — короткая пауза до 250 миллисекунд.
* medium — средняя пауза до 400 миллисекунд.
* strong — соответствует паузе после точки или предложения.
* x-strong — соответствует паузе после параграфа.
* none или x-weak — эти значения не добавляют паузу, а оставлены для совместимости с AWS API.
time	Длительность паузы в секундах или миллисекундах, например 2s или 400ms. Максимальная длительность паузы — 5 секунд.
При синтезе паузы указанной длительности может быть погрешность 100-200 миллисекунд.
<speak>Эй, секундочку<break time="1s"/> Что вы делаете?</speak>

Тег <break> добавляет паузу, даже если он идет после других элементов, добавляющих паузу, например после точки или запятой.

p
Используйте тег <p>, чтобы добавить паузу между абзацами. Пауза добавляется после закрывающего тега.

Пауза после абзаца больше, чем пауза после предложения или точки. Длительность паузы зависит от выбранного голоса, эмоциональной окраски, скорости и языка.

<speak>
  <p>Палач доказывал, что нельзя отрубить голову, у которой нет туловища, значит казнь не может состояться.</p>
  <p>Король доказывал, что все, имеющее голову, может быть обезглавлено, и что палач говорит пустяки.</p>
  <p>Королева тем временем вопила, что если кот не будет немедленно казнен, то казнены будут все присутствующие (замечание это удручающе подействовало на всех участников игры).</p>
</speak>

Все паузы внутри тега тоже учитываются. Например, на месте точки добавится дополнительная пауза, даже если она стоит перед закрывающим тегом.

phoneme
Используйте тег <phoneme>, чтобы контролировать правильность произношения с помощью фонем. Для воспроизведения будет использован текст, указанный в атрибуте ph. В атрибуте alphabet укажите используемый стандарт: ipa или x-sampa.

Международный фонетический алфавит (IPA)

<speak>
      В разных регионах России по-разному произносят букву
      <phoneme alphabet="ipa" ph="o">О</phoneme> в словах.
      Где-то говорят <phoneme alphabet="ipa" ph="məlɐko">молоко</phoneme>,
      а где-то <phoneme alphabet="ipa" ph="mələko">молоко</phoneme>.
</speak>

Extended Speech Assessment Methods Phonetic Alphabet (X-SAMPA)

<speak>
      В разных регионах России по-разному произносят букву
      <phoneme alphabet="x-sampa" ph="o">О</phoneme> в словах.
      Где-то говорят <phoneme alphabet="x-sampa" ph="m@l6ko">молоко</phoneme>,
      а где-то <phoneme alphabet="x-sampa" ph="m@l@ko">молоко</phoneme>.
</speak>

Список поддерживаемых фонем.

speak
Тег <speak> — это корневой тег. Весь текст должен быть внутри этого тега.

<speak>Мой текст в формате SSML.</speak>

s
Используйте тег <s>, чтобы добавить паузу между предложениями. Пауза после предложения равна паузе после точки. Длительность паузы зависит от выбранного голоса, эмоциональной окраски, скорости и языка.

<speak>
  <s>Первое предложение</s>
  <s>Второе предложение</s>
</speak>

Все паузы внутри тега тоже учитываются. Например на месте точки добавится дополнительная пауза, даже если она стоит перед закрывающим тегом.

sub
Используйте тег <sub>, чтобы подменить текст на другой текст при произношении. Например, чтобы правильно произнести аббревиатуру или название химического элемента.

<speak>
  Мой любимый химический элемент — <sub alias="ртуть">Hg</sub>, потому что блестит.
</speak>

Возвращай полный сценарий с добавленными SSML тегами.
"""
)
sound_chain = LLMChain(llm=llm, prompt=sound_prompt)


def write_ssml_to_file(ssml: str, filename: str = "text.xml"):

    with open(filename, "w", encoding="utf-8") as f:

        f.write(ssml)
    print(f"SSML-сценарий сохранен в файле: {filename}")


# 10. TextToSpeechAgent: generate audio file

def generate_audio(ssml):
    load_dotenv()
    url = 'https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize'

    api_key = os.getenv("API_KEY")
    folder_id = os.getenv("FOLDER_ID")
    print(api_key)
    print(folder_id)
    headers = {
        'Authorization': f'Api-Key {api_key}',
    }

    data = {
        'ssml': ssml,
        'lang': 'ru-RU',
        'voice': 'ermil',
        "emotion": "good",
        'folderId': folder_id,
    }

    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()

    with open('speech.ogg', 'wb') as out_file:
        out_file.write(response.content)


# Build a simple sequential pipeline
ingest_chain = SimpleSequentialChain(
    chains=[LLMChain(llm=llm, prompt=PromptTemplate(
        input_variables=["df"],
        template="""
Ingest articles from dataframe format, attach metadata, and output a JSON list of articles.

{df}
""")),
            ], verbose=True)

if __name__ == "__main__":
    # Example usage

    with open('article.txt', 'r', encoding='utf-8') as f:
        article_text = f.read()

    df = pd.DataFrame([{
        'id': 'article1',
        'text': article_text,
        'date': '1935-01-23',
        'author': 'Unknown',
        'section': 'News'
    }])

    articles = ingest_articles(df)
    print("Ingested articles:", articles)

    selected_ids = topic_chain.run(
        articles=articles,
        topic="Новости",
        date_range="1900-1950"
    )
    topic_temp = JsonOutputParser().parse(selected_ids)[0]
    topic_reasoning = topic_temp.get("reasoning")
    topic_array = topic_temp.get("array")
    print(f"Topic_picker_agent : {topic_reasoning}")
    selected = [a for a in articles if a["id"] in topic_array]

    summaries = {}
    for art in selected:
        summaries.update(summarize_article(art, levels=["short", "long"]))

    print("Summary_agent: Составленны длинные и краткие изложения статей", summaries)

    plan = segment_chain.run(summaries=summaries)
    plan_temp = JsonOutputParser().parse(plan)[0]
    plan_reasoning = plan_temp.get("reasoning")
    episode_plan = plan_temp.get("segments")
    print(f"Plan_agent : {plan_reasoning}")

    questions = question_chain.run(episode_plan=episode_plan, summaries=summaries)
    questions_temp = JsonOutputParser().parse(questions)[0]
    question_reasoning = questions_temp.get("reasoning")
    questions = questions_temp.get("questions")
    print(f"Question_maker_agent : {question_reasoning}")

    external_info = "none"
    script = script_chain.run(episode_plan=episode_plan, summaries=summaries, questions=questions,
                              external_info=external_info)
    script_temp = JsonOutputParser().parse(script)[0]
    script_reasoning = script_temp.get("reasoning")
    script_text = script_temp.get("script")
    print(f"Script_writer_agent : {script_reasoning}")

    qa = qa_chain.run(script_text=script_text)
    qa_temp = JsonOutputParser().parse(qa)[0]
    qa_reasoning = qa_temp.get("reasoning")
    qa_issues = qa_temp.get("issues")
    print(f"QA_agent : {qa_reasoning}")

    with open("script.txt", "w", encoding="utf-8") as f:
        f.write(script_text)

    if len(qa_issues) != 0:
        # Re-run script chain with fixes
        external_info = ""
        for issue in qa_issues:
            result = web_search(issue["description"])
            external_info += issue["description"]
            for found in result:
                external_info += str(found["title"]) + str(found["content"])
            external_info += "\n"
        edited_script = editor_chain.run(script_text=script_text, external_info=external_info)
        edited_script_temp = JsonOutputParser().parse(edited_script)[0]
        edited_script_reasoning = edited_script_temp.get("reasoning")
        edited_script_text = edited_script_temp.get("script")
        print(f"Editor_agent : {edited_script_reasoning}")

        with open("edited_script.txt", "w", encoding="utf-8") as f:
            f.write(edited_script_text)

        ssml = sound_chain.run(script_text=edited_script_text)
        with open("script_ssml.txt", "w", encoding="utf-8") as f:
            f.write(ssml)
        generate_audio(ssml)
    else:
        ssml = sound_chain.run(script_text=script_text)
        with open("script_ssml.txt", "w", encoding="utf-8") as f:
            f.write(ssml)
        generate_audio(ssml)