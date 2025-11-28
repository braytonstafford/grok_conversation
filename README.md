![xAI logo](https://brands.home-assistant.io/_/grok_conversation/icon.png)
# xAI Grok Conversation
This is a custom component of Home Assistant.

Derived from [OpenAI Conversation](https://www.home-assistant.io/integrations/openai_conversation/) but uses the xAI API URL [(https://api.x.ai/v1)](https://api.x.ai/v1).

## How it works
xAI Grok Conversation uses OpenAI's python package to call to call the xAI API URL to interact with the latest version of the Grok LLM, see xAI [documentation]([text](https://docs.x.ai/docs)).

## Installation
1. Install via registering as a custom repository of HACS or by copying `grok_conversation` folder into `<config directory>/custom_components`
2. Restart Home Assistant
3. Go to Settings > Devices & Services.
4. In the bottom right corner, select the Add Integration button.
5. Follow the instructions on screen to complete the setup (API Key is required).
    - [Generating an API Key](https://console.x.ai/)
6. Go to Settings > [Voice Assistants](https://my.home-assistant.io/redirect/voice_assistants/).
7. Click to edit Assistant (named "Home Assistant" by default).
8. Select "Grok" from "Conversation agent" tab.
    <details>

    <summary>Screenshot</summary>
    <img width="500" alt="Select a conversaion agent in Home Assistant" src="https://www.braytonstafford.com/content/images/2025/01/image-12.png">

    </details>

## Installation Walkthrough with Screenshots
A blog post walk-through on generating an API key, adding the custom repository, and setting up the integration can be [found here](https://braytonstafford.com/home-assistant-xai-grok-conversation-agent/)

## Preparation
After installed, you need to expose entities from "https://{your-home-assistant}/config/voice-assistants/expose" for the entities/devices to be controlled by your voice assistant.

## Configuration

### Enabling Device Control

To enable device control (turning lights on/off, controlling fans, etc.), you need to:

1. **Configure LLM HASS API**:
   - Go to Settings > Devices & Services > Grok Conversation
   - Click "Configure" on your Grok integration
   - Under "LLM HASS API", select your Home Assistant LLM API (not "No control")
   - If you don't have an LLM HASS API configured, you'll need to set one up first

2. **Expose Entities**:
   - Go to Settings > Voice Assistants > Expose entities
   - Select the entities you want Grok to be able to control
   - Entities must be exposed for Grok to interact with them

## Troubleshooting

### Tool Calls Not Working / Cannot Control Devices

If Grok cannot control devices or tool calls are not working:

1. **Check LLM HASS API Configuration**:
   - Verify that an LLM HASS API is selected in the integration options (not "No control")
   - Ensure the LLM HASS API integration is installed and configured
   - Check the Home Assistant logs for errors related to the LLM API

2. **Verify Entities Are Exposed**:
   - Go to Settings > Voice Assistants > Expose entities
   - Ensure the entities you want to control are selected
   - The integration will log a warning if no entities are exposed

3. **Check Logs**:
   - Enable debug logging by adding to `configuration.yaml`:
     ```yaml
     logger:
       logs:
         custom_components.grok_conversation: debug
     ```
   - Look for errors related to tool execution or LLM API calls

4. **Verify Integration Status**:
   - Check that the Grok Conversation integration shows as "Loaded" in Settings > Devices & Services
   - Restart Home Assistant if the integration shows errors

### Slow Response Times

If responses are taking too long:

1. **Use a Faster Model**:
   - Try `grok-4-fast` or `grok-2-beta` instead of `grok-4`
   - Configure in Settings > Devices & Services > Grok Conversation > Configure

2. **Reduce Max Tokens**:
   - Lower the max_tokens setting in the integration options
   - Default is 150, try reducing to 100 for faster responses

3. **Check API Status**:
   - Verify your xAI API key is valid and has sufficient credits
   - Check xAI API status page for any service issues

### "LLM HASS API not configured" Error

This error means:
- The LLM HASS API option is set to "none" or not configured
- You need to select an LLM HASS API in the integration options to enable device control
- If you don't have an LLM HASS API, you can still use Grok for conversations, but device control won't work

### Integration Breaks Home Assistant's OpenAI Integration

If you're experiencing conflicts with the official OpenAI integration:
- Ensure you're using different entity names or IDs
- Check that both integrations are using different API keys
- Consider using only one conversation agent at a time
