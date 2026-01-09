import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from langgraph.checkpoint.memory import MemorySaver
from ddgs import DDGS

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

@tool
def search(query: str) -> str:
    """Search the internet for latest information"""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        return " ".join([r["body"] for r in results])

tools = [search]

memory = MemorySaver()

agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=memory
)

print("Agent ready. Type exit to stop.")

thread_id = "demo_thread"

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": thread_id}}
    )

    msg = result["messages"][-1].content

    if isinstance(msg, list):
        print("Agent:", msg[0]["text"])
    else:
        print("Agent:", msg)