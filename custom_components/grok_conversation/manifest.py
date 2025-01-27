{
  "domain": "grok_conversation",
  "name": "xAI Grok Conversation",
  "version": "0.1.0",
  "after_dependencies": ["assist_pipeline", "intent"],
  "codeowners": ["@braytonstafford"],
  "config_flow": true,
  "dependencies": ["conversation"],
  "documentation": "https://github.com/braytonstafford/grok_conversation",
  "issue_tracker": "https://github.com/braytonstafford/grok_conversation/issues",
  "integration_type": "service",
  "iot_class": "cloud_polling",
  "requirements": ["openai==1.59.9"]
}