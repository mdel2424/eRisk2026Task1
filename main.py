import os
from langchain_openai import ChatOpenAI
from graph import app

# config for ollama (local hosting)
llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="llama3.2",
    temperature=0.2
)


def run_interaction():
    print("--- eRisk 2026: Conversational Depression Detection MVP ---")

    # init AgentState
    state = {
        "messages": [],
        "findings": [],
        "next_node": "somatic"
    }

    while True:
        user_input = input("\nPersona: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # append the user messages to state
        state["messages"].append({"role": "user", "content": user_input})

        # run the graph
        prediction = app.invoke(state)

        # update local state with graph results
        state = prediction

        print(f"\nAgent: {state['messages'][-1]['content']}")

        # debug (show current clinical findings)
        if state['findings']:
            print(f"--- Clinical Progress: {len(state['findings'])} symptoms logged ---")


if __name__ == "__main__":
    run_interaction()