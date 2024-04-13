from src.chatbot import ChatBot


def main():
    # Initialize the bot
    chatbot = ChatBot()

    # Main loop to interact with the bot
    while True:
        user_input = input("You: ")  # Get user input
        if user_input.lower() == 'exit':
            print("Bot: Goodbye!")
            break
        else:
            # Process user input and get bot response
            bot_response = chatbot.respond(user_input)
            print("Bot:", bot_response)


if __name__ == "__main__":
    main()
