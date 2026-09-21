from autogeo.utils.openrouter import call_openrouter


models = [
    "gemini-2.5-flash-lite",
    "gpt-4o-mini",
    "claude-3-haiku-20240307",
]


for model in models:

    print("=" * 60)
    print("Testing:", model)

    try:
        result = call_openrouter(
            user_prompt="Reply exactly with: API OK",
            model_name=model,
            temperature=0
        )

        print("Response:", result)

    except Exception as e:
        print("ERROR:", e)