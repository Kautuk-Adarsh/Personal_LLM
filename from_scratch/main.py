# main.py
from from_scratch.model import LLM

model= LLM("Hello my name is kautuk adarsh",10 , 5)
output =model.forward("Hello my name is kautuk")
print(output)