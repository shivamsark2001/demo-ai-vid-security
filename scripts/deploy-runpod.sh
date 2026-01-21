#!/bin/bash

# Deploy RunPod Worker Script
# Run this after building and pushing the Docker image

RUNPOD_API_KEY="${RUNPOD_API_KEY:?Please set RUNPOD_API_KEY environment variable}"
DOCKER_IMAGE="${1:-shivamsark2001/video-security-worker:latest}"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?Please set OPENROUTER_API_KEY environment variable}"

echo "🚀 Creating RunPod Serverless Endpoint..."
echo "   Docker Image: $DOCKER_IMAGE"

# Create the endpoint via GraphQL
RESPONSE=$(curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"mutation { saveEndpoint(input: { name: \\\"video-security-analyzer\\\", imageName: \\\"$DOCKER_IMAGE\\\", gpuIds: \\\"AMPERE_24\\\", env: [{key: \\\"OPENROUTER_API_KEY\\\", value: \\\"$OPENROUTER_API_KEY\\\"}], scalerType: \\\"QUEUE_DELAY\\\", scalerValue: 4, workersMin: 0, workersMax: 3, idleTimeout: 5 }) { id name } }\"
  }")

echo "Response: $RESPONSE"

# Extract endpoint ID
ENDPOINT_ID=$(echo $RESPONSE | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)

if [ -n "$ENDPOINT_ID" ]; then
  echo ""
  echo "✅ Endpoint Created!"
  echo "   Endpoint ID: $ENDPOINT_ID"
  echo ""
  echo "📝 Next Steps:"
  echo "   1. Update Vercel env:"
  echo "      printf '$ENDPOINT_ID' | npx vercel env add RUNPOD_ENDPOINT_ID production --force"
  echo "   2. Redeploy:"
  echo "      npx vercel --prod --yes"
else
  echo ""
  echo "❌ Failed to create endpoint. Check the response above."
  echo ""
  echo "💡 Manual setup:"
  echo "   1. Go to https://www.runpod.io/console/serverless"
  echo "   2. Create new endpoint"
  echo "   3. Use image: $DOCKER_IMAGE"
  echo "   4. Add env: OPENROUTER_API_KEY"
  echo "   5. Copy endpoint ID and update Vercel"
fi
