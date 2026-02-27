from graph import build_graph
from IPython.display import Image, display
graph = build_graph()

while True:

    user_input = input("\nVous: ")

    state = {
        "user_input": user_input
    }

    result = graph.invoke(state)
    display(Image(graph.get_graph().draw_mermaid_png(), width=800))
    print("\nAssistant:")
    print(result["response"])