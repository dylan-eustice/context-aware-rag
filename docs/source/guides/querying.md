# Querying Documents

This guide explains how to query documents in the Context-Aware RAG system.

## Making Queries

Queries can be made to the system using the `/call` endpoint of the Retrieval Service.

### Request Format

```json
{
  "state": {
    "chat": {
      "question": "Your question here",
      "is_live": false,
    }
  }
}
```

### Example Query

```python
import requests

url = "http://localhost:8000/call"
headers = {"Content-Type": "application/json"}
data = {
    "state": {
        "chat": {
            "question": "What topics are covered in the document?",
            "is_live": False,
        }
    }
}

response = requests.post(url, headers=headers, json=data)
print(response.text)
```

### Query Parameters

- `question`: The actual question you want to ask about the documents
- `is_live`: Set to `true` for real-time queries, `false` for batch processing

## Best Practices

1. **Question Formulation**
   - Be specific and clear in your questions
   - Use natural language
   - Avoid overly complex or multi-part questions

2. **Query Timing**
   - For real-time applications, set `is_live: true`
   - For batch processing, set `is_live: false`

3. **Error Handling**
   - Always check response status codes
   - Handle timeouts appropriately
   - Implement retry logic for failed requests
