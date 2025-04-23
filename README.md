# ExploreEase: A Multi-Agent Conversational AI System for Intelligent Travel Planning

ExploreEase is an innovative, AI-powered travel planning application that delivers **personalized travel recommendations** through a **conversational chatbot interface**. Built using **Streamlit** and powered by the **TaskflowAI** framework, the system orchestrates multiple intelligent agents to help users explore destinations, research activities, and find real-time flight options.

This project was developed as part of a **Master of Science in Data Science at SVKM’s NMIMS University** and is detailed in the thesis titled:  
> *"ExploreEase: A Multi-Agent Conversational AI System for Intelligent Travel Planning"*

---

## 🌟 Features

- **💬 Conversational Interface**  
  Streamlit-based chatbot interface for natural, intuitive travel planning.

- **🤖 Multi-Agent Architecture**  
  Powered by TaskflowAI with specialized agents for:
  - Destination Research
  - Parameter Extraction
  - Missing Parameter Detection
  - Parameter Combination
  - Flight Search

- **🗺️ Personalized Destination Research**  
  Travel reports with tailored activities, attractions, and images using Wikipedia.

- **✈️ Automated Flight Search**  
  Real-time flight options retrieved from Amadeus API based on validated user input.

- **🧠 Memory Retention**  
  Stores past interactions and results using a Faiss Knowledge Base for context-aware experiences.

- **🔧 Modular & Scalable**  
  Designed for performance, easy extension, and future upgrades.

- **🧠 LLM Integration**  
  Uses OpenRouter (Deepseek V3) and Groq (Deepseek R1) for robust natural language processing.

---

## 📈 Performance Highlights

- ✅ **Efficiency:** 40% reduction in token usage compared to monolithic design.
- ⚡ **Speed:** Reduced processing times to as low as 4 minutes 20 seconds using Gemma2 9b-it LLM.
- 💸 **Cost Savings:** Session costs dropped from ~$0.021 (monolithic) to $0.010–$0.013 (multi-agent).
- 🧠 **Improved UX:** Modular design enabled better user control, faster iteration, and higher satisfaction.

---

## 🔮 Future Enhancements

- 🏨 **Hotel Recommendations**  
  Integrate hotel booking APIs for real-time accommodation options.

- 🗓️ **Itinerary Generation**  
  Implement day-by-day travel plans using CrewAI with customizable schedules.

- 🖼️ **Image-Based Suggestions**  
  Add computer vision to suggest destinations based on uploaded travel photos.

- 🧠 **Domain-Specific LLMs**  
  Fine-tune models for tourism to improve the relevance and quality of recommendations.

---

## 📋 Usage

1. **Launch the App:** Access the chatbot interface in your browser.
2. **Explore Destinations:**  
   Input queries like `"Tell me about Bali for adventure lovers"`  
   → Get tailored activities, attractions, and images.
3. **Search Flights:**  
   Input: `"Flights from Mumbai to Bali for 3 adults, 5th May to 14th May 2025, economy, INR, non-stop"`  
   → Missing details? The system prompts you!
4. **Regenerate or Iterate:**  
   Try new destinations or tweak your preferences.
