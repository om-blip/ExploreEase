import streamlit as st
import os
from dotenv import load_dotenv
from taskflowai import GroqModels, Agent, OpenrouterModels, Task, AmadeusTools, WikipediaTools
from taskflowai.knowledgebases import FaissKnowledgeBase

# Load environment variables
load_dotenv()

weather_api_key = os.getenv("WEATHER_API_KEY")
amadeus_api_key = os.getenv("AMADEUS_API_KEY")
amadeus_api_secret = os.getenv("AMADEUS_API_SECRET")
groq_api_key = os.getenv("GROQ_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")


current_dir = os.path.dirname(os.path.abspath(__file__))
faiss_save_path = os.path.join(current_dir, "my_faiss_index")

faiss_kb = FaissKnowledgeBase(
    kb_name="my_knowledgebase",
    embedding_provider="cohere",
    embedding_model="embed-english-v3.0",
    save_to_filepath=faiss_save_path,
    api_key=cohere_api_key
)


# Define Agents
web_research_agent = Agent(
    role="Web Research Agent",
    goal="Extract the destination and interest of the traveller from the user query and do research about the destination and find relevant images",
    attributes="diligent, thorough, comprehensive, visual-focused",
    llm=OpenrouterModels.custom_model(model_name='deepseek/deepseek-chat-v3-0324:free'),
    tools=[WikipediaTools.search_images, WikipediaTools.search_articles]
)

parameter_agent = Agent(
    role='Assistant',
    goal=('Extract the location, destination, departure date, return date, travel class, number of adult tickets, number of children ticket, max price, non_stop, currency from a user query and create a goal for\n'
          'another agent specifying the identified parameters in the goal to find 3 flights for their trip. Also do not make any assumptions about the parameters if it is not specified. Note that the\n'
          'the meaning of the parameter non_stop is that the user wants direct flight or flight that have haults.'),
    attributes='Efficient and hardworking',
    llm=GroqModels.custom_model(model_name="deepseek-r1-distill-llama-70b"),
)
missing_parameter_agent = Agent(
    role='Assistant',
    goal=('The tool for searching flight requires following parameters location, destination, departure date, return data, travel class, number of adult tickets, nummber of children ticket, max price, non_stop, currency.\n'
          ' You will be provided a query and you will find if any above mentioned parameters are missing. If any parameters are missing you will create a query asking the user to specify those parameters.\n'
          'Always start your query with yes if parameter are missing and no if parameters are not missing.'),
    attributes='Concise and Efficient',
    llm=GroqModels.custom_model(model_name="deepseek-r1-distill-llama-70b"),
)
parameters_combining_agent = Agent(
    role='Assistant',
    goal=('You will be provided with two querys having different parameters for searching flight tickets. There might be some missing parameters in first query which will be present in \n'
          'the second query. You have to combine the parameters from both the queries and create a goal for another agent specifying the identified parameters in the goal to find 3 flights\n'
          'for their trip.'),
    attributes='Efficient and hardworking',
    llm=GroqModels.custom_model(model_name="deepseek-r1-distill-llama-70b"),
)
travel_agent = Agent(
    role="Travel Agent",
    goal="Assist travelers with to find 3 flights for their destinations",
    attributes="friendly, hardworking, and efficient in reporting back to users",
    llm=OpenrouterModels.custom_model(model_name='deepseek/deepseek-chat-v3-0324:free'),
    tools=[AmadeusTools.search_flights],
    max_tokens=2000
)

# Backend Functions
def research_destination(query):
    instruction = (
        f"Create a comprehensive report about destination with the following:\n"
        f"1. Use Wikipedia tools to find and include 2-3 high-quality images of key attractions\n"
        f"2. Ensure images are full URLs starting with http:// or https://\n"
        f"3. Format images as: ![Description](https://full-image-url)\n"
        f"4. Include a brief caption for each image\n"
        f"5. Research attractions and activities related to: interests\n"
        f"6. Organize the report with proper headings and sections\n"
        f"7. Place images naturally throughout the content where relevant\n"
        f"8. Include practical visitor information\n"
        f"Format the entire response in clean markdown"
    )
    return Task.create(
        agent=web_research_agent,
        context=f"User Query: {query}",
        instruction=instruction
    )

def search_parameter(query):
    return Task.create(
        agent=parameter_agent,
        context=f"Extract the required parameters from the following User query: {query}",
        instruction="Be perfect in your response."
    )

def missing_parameter(query):
    return Task.create(
        agent=missing_parameter_agent,
        context=f"Identify missing parameters from the following User query: {query}",
        instruction="Be perfect in your response."
    )

def parameter_combiner(query1, query2):
    return Task.create(
        agent=parameters_combining_agent,
        context=f"First query: {query1}\nSecond query: {query2}",
        instruction="Be perfect in your response."
    )

def search_flights(query):
    return Task.create(
        agent=travel_agent,
        context=query,
        instruction="Find top 3 convenient flight options and provide concise bullet point information."
    )

st.title("Travel Planner Chatbot")

# Initialize session state for chat and content storage
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'stage' not in st.session_state:
    st.session_state.stage = 'input'
if 'flight_search_stage' not in st.session_state:
    st.session_state.flight_search_stage = None
if 'user_query' not in st.session_state:
    st.session_state.user_query = None
if 'destination_report' not in st.session_state:
    st.session_state.destination_report = None
if 'flight_options' not in st.session_state:
    st.session_state.flight_options = None
if 'combined_flight_parameters' not in st.session_state:
    st.session_state.combined_flight_parameters = None

# Chat container
chat_container = st.container()

# Display chat history
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Chat input at the bottom
user_input = st.chat_input("How can I assist you with your travel plans today?")

def add_message(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})

def process_destination_response(response):
    marker = "✔ Request completed"
    last_index = str(response).rfind(marker)
    return str(response)[last_index + len(marker):].strip() if last_index != -1 else str(response).strip()

def process_flight_response(response):
    marker = "</think"
    last_index = str(response).rfind(marker)
    return str(response)[last_index + len(marker):].strip() if last_index != -1 else str(response).strip()

# Handle user input
if user_input:
    # Add user message to chat
    add_message("user", user_input)

    if st.session_state.stage == 'input':
        with st.chat_message("assistant"):
            with st.spinner("Planning your trip..."):
                response = research_destination(user_input)
                report = process_destination_response(response)
                
                if report:
                    st.write("Here's your travel plan:")
                    st.write(report)
                    st.write("What would you like to do next?\n1. Search for flights\n2. Plan another destination\n3. Regenerate this travel plan")
                    faiss_kb.add_memory(report)
                    st.session_state.user_query = user_input
                    st.session_state.destination_report = report
                    st.session_state.stage = 'results'
                    add_message("assistant", f"Here's your travel plan:\n{report}\n\nWhat would you like to do next?\n1. Search for flights\n2. Plan another destination\n3. Regenerate this travel plan")
                else:
                    st.error("Couldn't generate a travel plan. Please try again.")
                    add_message("assistant", "Sorry, I couldn't generate a travel plan. Please try again with more details.")

    elif st.session_state.stage == 'results':
        with st.chat_message("assistant"):
            if "flight" in user_input.lower() or "flights" in user_input.lower() or "1" in user_input.lower():
                st.session_state.stage = 'flights_hotels'
                st.session_state.flight_search_stage = 'initial'
                st.write("Great! Let's find some flights. Please tell me your flight preferences (e.g., departure city, dates, number of passengers, etc.)")
                add_message("assistant", "Great! Let's find some flights. Please tell me your flight preferences (e.g., departure city, dates, number of passengers, etc.)")
            elif "another" in user_input.lower() or "destination" in user_input.lower() or "2" in user_input.lower():
                st.session_state.stage = 'input'
                st.session_state.user_query = None
                st.session_state.flight_search_stage = None
                st.session_state.destination_report = None
                st.session_state.flight_options = None
                st.session_state.combined_flight_parameters = None
                st.write("Sure, let's plan a new trip! Where would you like to go?")
                add_message("assistant", "Sure, let's plan a new trip! Where would you like to go?")
            elif "regenerate" in user_input.lower() or "travel plan" in user_input.lower() or "3" in user_input.lower():
                if st.session_state.user_query and st.session_state.destination_report:
                    with st.spinner("Regenerating your travel plan..."):
                        response = research_destination(st.session_state.user_query)
                        new_report = process_destination_response(response)
                        if new_report:
                            st.write("Here's your regenerated travel plan:")
                            st.write(new_report)
                            st.write("What would you like to do next?\n1. Search for flights\n2. Plan another destination\n3. Regenerate this travel plan")
                            faiss_kb.add_memory(new_report)
                            st.session_state.destination_report = new_report
                            add_message("assistant", f"Here's your regenerated travel plan:\n{new_report}\n\nWhat would you like to do next?\n1. Search for flights\n2. Plan another destination\n3. Regenerate this travel plan")
                        else:
                            st.error("Couldn't regenerate the travel plan. Please try again.")
                            add_message("assistant", "Sorry, I couldn't regenerate the travel plan. Please try again.")
                else:
                    st.write("No previous travel plan to regenerate. Please create a new one.")
                    add_message("assistant", "No previous travel plan to regenerate. Please create a new one.")
            else:
                st.write("I didn't quite understand. What would you like to do?\n1. Search for flights\n2. Plan another destination\n3. Regenerate this travel plan")
                add_message("assistant", "I didn't quite understand. What would you like to do?\n1. Search for flights\n2. Plan another destination\n3. Regenerate this travel plan")

    elif st.session_state.stage == 'flights_hotels':
        with st.chat_message("assistant"):
            with st.spinner("Processing your flight request..."):
                if st.session_state.flight_search_stage == 'initial':
                    st.session_state.user_query = user_input
                    parameters = search_parameter(user_input)
                    missing_parameters = missing_parameter(parameters)

                    if 'yes' in str(missing_parameters).lower():
                        st.write(missing_parameters)
                        st.session_state.flight_search_stage = 'missing'
                        add_message("assistant", str(missing_parameters) + "\nPlease provide the missing details.")
                    else:
                        flights_report = search_flights(parameters)
                        flight_info = process_flight_response(flights_report)
                        st.write("Here are your flight options:")
                        st.write(flight_info)
                        st.write("What would you like to do next?\n1. Search for different flights\n2. Return to destination planning\n3. Regenerate these flight options")
                        faiss_kb.add_memory(flight_info)
                        st.session_state.flight_options = flight_info
                        st.session_state.combined_flight_parameters = parameters  # Store initial parameters
                        st.session_state.flight_search_stage = 'results'
                        add_message("assistant", f"Here are your flight options:\n{flight_info}\n\nWhat would you like to do next?\n1. Search for different flights\n2. Return to destination planning\n3. Regenerate these flight options")

                elif st.session_state.flight_search_stage == 'missing':
                    combined_parameters = parameter_combiner(st.session_state.user_query, user_input)
                    flights_report = search_flights(combined_parameters)
                    flight_info = process_flight_response(flights_report)
                    st.write("Here are your flight options:")
                    st.write(flight_info)
                    st.write("What would you like to do next?\n1. Search for different flights\n2. Return to destination planning\n3. Regenerate these flight options")
                    faiss_kb.add_memory(flight_info)
                    st.session_state.flight_options = flight_info
                    st.session_state.combined_flight_parameters = combined_parameters  # Store combined parameters
                    st.session_state.flight_search_stage = 'results'
                    add_message("assistant", f"Here are your flight options:\n{flight_info}\n\nWhat would you like to do next?\n1. Search for different flights\n2. Return to destination planning\n3. Regenerate these flight options")

                elif st.session_state.flight_search_stage == 'results':
                    if "regenerate" in user_input.lower() or "flight options" in user_input.lower() or "3" in user_input.lower():
                        if st.session_state.combined_flight_parameters and st.session_state.flight_options:
                            with st.spinner("Regenerating your flight options..."):
                                flights_report = search_flights(st.session_state.combined_flight_parameters)
                                new_flight_info = process_flight_response(flights_report)
                                if new_flight_info:
                                    st.write("Here are your regenerated flight options:")
                                    st.write(new_flight_info)
                                    st.write("What would you like to do next?\n1. Search for different flights\n2. Return to destination planning\n3. Regenerate these flight options")
                                    faiss_kb.add_memory(new_flight_info)
                                    st.session_state.flight_options = new_flight_info
                                    add_message("assistant", f"Here are your regenerated flight options:\n{new_flight_info}\n\nWhat would you like to do next?\n1. Search for different flights\n2. Return to destination planning\n3. Regenerate these flight options")
                                else:
                                    st.error("Couldn't regenerate flight options. Please try again.")
                                    add_message("assistant", "Sorry, I couldn't regenerate flight options. Please try again.")
                        else:
                            st.write("No previous flight options to regenerate. Please provide a new flight query.")
                            add_message("assistant", "No previous flight options to regenerate. Please provide a new flight query.")
                    elif "different" in user_input.lower() or "flights" in user_input.lower() or "1" in user_input.lower():
                        st.session_state.flight_search_stage = 'initial'
                        st.session_state.combined_flight_parameters = None  # Clear combined parameters for new search
                        st.write("Please provide your new flight preferences.")
                        add_message("assistant", "Please provide your new flight preferences.")
                    elif "destination" in user_input.lower() or "planning" in user_input.lower() or "2" in user_input.lower():
                        st.session_state.stage = 'input'
                        st.session_state.user_query = None
                        st.session_state.flight_search_stage = None
                        st.session_state.flight_options = None
                        st.session_state.combined_flight_parameters = None
                        st.write("Sure, let's plan a new trip! Where would you like to go?")
                        add_message("assistant", "Sure, let's plan a new trip! Where would you like to go?")
                    else:
                        st.write("I didn't quite understand. What would you like to do?\n1. Search for different flights\n2. Return to destination planning\n3. Regenerate these flight options")
                        add_message("assistant", "I didn't quite understand. What would you like to do?\n1. Search for different flights\n2. Return to destination planning\n3. Regenerate these flight options")