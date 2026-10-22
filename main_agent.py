
from agent.brain import ResearchAgent


def main():
    print("=== Autonomous Research Agent ===")
    print("Commands: reset | exit\n")

    agent = ResearchAgent()

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if question.lower() == "reset":
            agent.reset()
            print("Conversation cleared.\n")
            continue

        if not question:
            continue

        try:
            answer = agent.run(question)
            print(f"\nAgent: {answer}\n")

        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()
