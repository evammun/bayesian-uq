import requests, json, math

resp = requests.post("http://localhost:11434/api/chat", json={
    "model": "qwen3:8b-q4_K_M",
    "messages": [{"role": "user", "content": (
        "What is the primary function of mitochondria?\n\n"
        "  A) Protein synthesis\n"
        "  B) Energy production\n"
        "  C) Cell division\n"
        "  D) DNA replication\n\n"
        "Answer with ONLY the letter (A, B, C, or D)."
    )}],
    "stream": False,
    "think": False,
    "logprobs": True,
    "top_logprobs": 20,
    "options": {"temperature": 0.7, "num_predict": 1},
})

data = resp.json()
print("Response:", data.get("message", {}).get("content"))
print()

# Look at the first token's logprobs
if data.get("logprobs"):
    first_token = data["logprobs"][0]
    print(f"Selected token: {first_token['token']} (logprob: {first_token['logprob']:.4f})")
    print()
    
    # Extract probabilities for A, B, C, D
    answer_probs = {}
    for tp in first_token.get("top_logprobs", []):
        token = tp["token"].strip()
        if token in ["A", "B", "C", "D"]:
            answer_probs[token] = math.exp(tp["logprob"])
    
    # Normalise over just A/B/C/D
    total = sum(answer_probs.values())
    if total > 0:
        print("Answer distribution (normalised over A/B/C/D):")
        for letter in ["A", "B", "C", "D"]:
            prob = answer_probs.get(letter, 0) / total
            print(f"  {letter}: {prob:.4f} ({prob*100:.1f}%)")