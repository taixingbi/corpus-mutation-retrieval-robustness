SERVICE_URL=$(aws cloudformation describe-stacks \
  --region us-east-1 \
  --stack-name ecs-reranker-mvp \
  --query "Stacks[0].Outputs[?OutputKey=='ServiceUrl'].OutputValue" \
  --output text)

API_KEY=1234
# One ECS task hosts all models; request "model" selects which CrossEncoder to run.
# First request for a model may be slow (lazy load).

curl -sS "${SERVICE_URL}/health" -H "Authorization: Bearer ${API_KEY}" | jq .
echo
curl -sS "${SERVICE_URL}/v1/models" -H "Authorization: Bearer ${API_KEY}" | jq .
echo

QUERY='what is a panda?'
DOCS='[
  "The giant panda is a bear species endemic to China.",
  "Pandas is a Python library for data analysis.",
  "Bamboo is the primary food of giant pandas."
]'

# cross-encoder/ms-marco-MiniLM-L6-v2 (22M)
curl -sS -X POST "${SERVICE_URL}/v1/rerank" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "{
    \"model\": \"cross-encoder/ms-marco-MiniLM-L6-v2\",
    \"query\": \"${QUERY}\",
    \"documents\": ${DOCS},
    \"top_n\": 2,
    \"return_documents\": true
  }" | jq '{model, results, error, detail}'
echo

# jinaai/jina-reranker-v2-base-multilingual
curl -sS -X POST "${SERVICE_URL}/v1/rerank" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "{
    \"model\": \"jinaai/jina-reranker-v2-base-multilingual\",
    \"query\": \"${QUERY}\",
    \"documents\": ${DOCS},
    \"top_n\": 2,
    \"return_documents\": true
  }" | jq '{model, results, error, detail}'
echo

# mixedbread-ai/mxbai-rerank-large-v1
curl -sS -X POST "${SERVICE_URL}/v1/rerank" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "{
    \"model\": \"mixedbread-ai/mxbai-rerank-large-v1\",
    \"query\": \"${QUERY}\",
    \"documents\": ${DOCS},
    \"top_n\": 2,
    \"return_documents\": true
  }" | jq '{model, results, error, detail}'
echo

# BAAI/bge-reranker-v2-m3 (default)
curl -sS -X POST "${SERVICE_URL}/v1/rerank" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "{
    \"model\": \"BAAI/bge-reranker-v2-m3\",
    \"query\": \"${QUERY}\",
    \"documents\": ${DOCS},
    \"top_n\": 2,
    \"return_documents\": true
  }" | jq '{model, results, error, detail}'
echo
