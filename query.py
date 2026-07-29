from fdsn_agent import Agent, LLMConfig

# Local Ollama
cfg   = LLMConfig.from_preset("ollama", model="gemma4")
agent = Agent(cfg)
result = agent.query("Stations within 5 km of Anchorage, Alaska")

print(result.summary)
# print(result.data["count"])
# print(result.to_json(pretty=True))
# print(result.data.to_json(pretty=True))

