from fastapi import FastAPI
from pydantic import BaseModel
from retriever import *

app = FastAPI()

retriever = Retriever(path_saved_model="https://huggingface.co/xValentim/vector-search-bert-based/resolve/main/vae_model_state_dict_2.pth")

@app.get("/")
def read_root():
    return {"Status": "Running..."}

@app.get("/query")
def query(input_content: str):
    output = retriever.query(input_content, k=10)
    output_filtro = []
    for x in output:
        if x.score > 0.985:
            output_filtro.append(x)
    return {
        "results": output_filtro, 
        "message": "OK"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=1414)