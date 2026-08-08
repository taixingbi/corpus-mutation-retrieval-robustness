FUNCTION_URL=$(aws cloudformation describe-stacks \
  --region us-east-1 \
  --stack-name bedrock-inference-mvp \
  --query "Stacks[0].Outputs[?OutputKey=='InferenceFunctionUrl'].OutputValue" \
  --output text)
INFERENCE_API_KEY=1234


curl -sS -X POST "${FUNCTION_URL}v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${INFERENCE_API_KEY}" \
  -d '{
    "model": "openai.gpt-oss-120b-1:0",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }' | jq '{error, detail, model, answer: .choices[0].message.content, usage}'
echo