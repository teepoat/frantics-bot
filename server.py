from fastapi import FastAPI
from pydantic import BaseModel
from models.transformer.text_generator import TextGenerator


app = FastAPI()

generator = TextGenerator(
    model_name='fine_tuned_model_gpt_2',
)


class Message(BaseModel):
    author: str
    content: str


@app.post("/generate")
def generate_response(message: Message):
    response = generator.generate_text(
        author=message.author,
        input_str=message.content,
        max_length=100,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,  # Слегка уменьшаем уверенность
        top_k=100,         # Уменьшаем количество рассматриваемых верхних k слов
        top_p=0.95        # Уменьшаем "ядерность" распределения
    )["generated_texts"][0]
    response = response[:response.find("</s>")]
    return { "response": response }
