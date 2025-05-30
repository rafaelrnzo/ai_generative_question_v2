import ollama

# Make sure your client machine can reach 10.50.0.101
# and that Ollama is listening on that IP on the server.
client = ollama.Client(host='http://10.50.0.101:11434')

try:
    response = client.chat(model='llama3.2:latest', messages=[{'role': 'user', 'content': 'Hello, Ollama!'}])
    print(response['message']['content'])
except Exception as e:
    print(f"Error connecting to Ollama: {e}")