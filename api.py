from huggingface_hub import login
from dotenv import load_dotenv
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from module.functions import pipeline
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

load_dotenv()
TOKEN = os.environ['access']
login(token=TOKEN)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate-image/")
async def generate_image(request: ImageRequest):
    # Here you would call the pipeline function to generate the image
    image = pipeline(request.prompt)
    return {"image": image}


@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse(
        request=request, name="item.html", context={"id": id}
    )

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request, name="prompt.html"
    )