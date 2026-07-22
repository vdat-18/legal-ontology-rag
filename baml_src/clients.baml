// 9Router exposes an OpenAI-compatible API while BAML owns prompting and parsing.
client<llm> Router {
  provider "openai-generic"
  options {
    base_url env.LLM_BASE_URL
    api_key env.LLM_API_KEY
    model env.LLM_MODEL
  }
}
