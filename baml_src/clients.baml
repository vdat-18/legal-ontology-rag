// The runtime provider exposes an OpenAI-compatible API while BAML owns prompting and parsing.
client<llm> OpenAICompatibleProvider {
  provider "openai-generic"
  options {
    base_url env.LLM_BASE_URL
    api_key env.LLM_API_KEY
    model env.LLM_MODEL
    supports_streaming false
    reasoning_effort "none"
  }
}
