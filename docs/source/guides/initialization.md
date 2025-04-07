# Service Initialization Guide

This guide explains how to initialize the Context-Aware RAG services.

## Service Initialization

Both services must be initialized before use with the same UUID to ensure proper communication.

### Initialize Data Ingestion Service

```python
import requests

url = "http://localhost:8001/init"
headers = {
    "Content-Type": "application/json"
}
data = {
    "config_path": "/app/config/config.yaml",
    "uuid": "your_session_uuid"
}

response = requests.post(url, headers=headers, json=data)
print(response.text)
```

### Initialize Retrieval Service

```python
import requests

url = "http://localhost:8000/init"
headers = {
    "Content-Type": "application/json"
}
data = {
    "config_path": "/app/config/config.yaml",
    "uuid": "your_session_uuid"
}

response = requests.post(url, headers=headers, json=data)
print(response.text)
```

**Important**: Use the same UUID for both services to ensure they can access the same context.

## UUID Management

### Best Practices

1. **UUID Generation**
   - Use unique UUIDs for different sessions
   - Consider using timestamp-based UUIDs for easy tracking
   - Avoid reusing UUIDs across different contexts

2. **UUID Consistency**
   - Always use the same UUID for both services
   - Store the UUID securely if needed for later use
   - Document UUID usage in your application

3. **UUID Security**
   - Treat UUIDs as sensitive information
   - Avoid exposing UUIDs in logs or error messages
   - Implement proper UUID validation

## Configuration

The initialization process requires a configuration file at `/app/config/config.yaml`. This file controls various aspects of the system:

- Vector store settings
- Model parameters
- Chunking configuration
- Logging settings

Make sure the configuration file is properly set up before initializing the services.
