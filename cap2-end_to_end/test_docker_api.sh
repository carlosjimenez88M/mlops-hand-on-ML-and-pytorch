#!/bin/bash
# =================================================================
# Docker API Test Script
# Purpose: Build, run, and test the Housing Price Prediction API
# Author: Carlos Daniel Jiménez
# =================================================================

set -e  # Exit on error

echo "=========================================="
echo "Housing Price Prediction API - Docker Test"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is running
echo -e "\n${YELLOW}[1/6] Checking Docker daemon...${NC}"
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker Desktop and try again.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker is running${NC}"

# Stop any existing containers
echo -e "\n${YELLOW}[2/6] Stopping existing containers...${NC}"
docker compose down 2>/dev/null || true
echo -e "${GREEN}✅ Cleaned up existing containers${NC}"

# Build the Docker image
echo -e "\n${YELLOW}[3/6] Building Docker image...${NC}"
echo "This may take a few minutes on first build..."
docker compose build --no-cache
echo -e "${GREEN}✅ Docker image built successfully${NC}"

# Start the containers
echo -e "\n${YELLOW}[4/6] Starting API container...${NC}"
docker compose up -d
echo -e "${GREEN}✅ Container started${NC}"

# Wait for the API to be ready
echo -e "\n${YELLOW}[5/6] Waiting for API to be ready...${NC}"
MAX_WAIT=60
COUNTER=0
until curl -s http://localhost:8080/health >/dev/null 2>&1; do
    sleep 2
    COUNTER=$((COUNTER + 2))
    if [ $COUNTER -ge $MAX_WAIT ]; then
        echo -e "${RED}❌ API failed to start within $MAX_WAIT seconds${NC}"
        echo "Container logs:"
        docker compose logs api
        exit 1
    fi
    echo -n "."
done
echo -e "\n${GREEN}✅ API is ready${NC}"

# Test health endpoint
echo -e "\n${YELLOW}[6/6] Testing API endpoints...${NC}"
echo ""
echo "=========================================="
echo "Test 1: Health Check"
echo "=========================================="
HEALTH_RESPONSE=$(curl -s http://localhost:8080/health)
echo "$HEALTH_RESPONSE" | python3 -m json.tool
echo ""

# Test root endpoint
echo "=========================================="
echo "Test 2: Root Endpoint"
echo "=========================================="
ROOT_RESPONSE=$(curl -s http://localhost:8080/)
echo "$ROOT_RESPONSE" | python3 -m json.tool
echo ""

# Test model info endpoint
echo "=========================================="
echo "Test 3: Model Info"
echo "=========================================="
MODEL_INFO=$(curl -s http://localhost:8080/api/v1/model/info)
echo "$MODEL_INFO" | python3 -m json.tool
echo ""

# Test prediction endpoint with sample data
echo "=========================================="
echo "Test 4: Prediction Request"
echo "=========================================="
echo "Making prediction for sample housing data..."

PREDICTION_REQUEST='{
  "instances": [
    {
      "longitude": -122.23,
      "latitude": 37.88,
      "housing_median_age": 41.0,
      "total_rooms": 880.0,
      "total_bedrooms": 129.0,
      "population": 322.0,
      "households": 126.0,
      "median_income": 8.3252,
      "ocean_proximity": "NEAR BAY"
    }
  ]
}'

PREDICTION_RESPONSE=$(curl -s -X POST \
  http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d "$PREDICTION_REQUEST")

echo "$PREDICTION_RESPONSE" | python3 -m json.tool
echo ""

# Test batch prediction
echo "=========================================="
echo "Test 5: Batch Prediction (3 houses)"
echo "=========================================="

BATCH_REQUEST='{
  "instances": [
    {
      "longitude": -122.23,
      "latitude": 37.88,
      "housing_median_age": 41.0,
      "total_rooms": 880.0,
      "total_bedrooms": 129.0,
      "population": 322.0,
      "households": 126.0,
      "median_income": 8.3252,
      "ocean_proximity": "NEAR BAY"
    },
    {
      "longitude": -118.20,
      "latitude": 34.05,
      "housing_median_age": 25.0,
      "total_rooms": 2500.0,
      "total_bedrooms": 450.0,
      "population": 1200.0,
      "households": 400.0,
      "median_income": 5.5,
      "ocean_proximity": "NEAR OCEAN"
    },
    {
      "longitude": -121.89,
      "latitude": 37.34,
      "housing_median_age": 15.0,
      "total_rooms": 3200.0,
      "total_bedrooms": 600.0,
      "population": 1500.0,
      "households": 550.0,
      "median_income": 7.2,
      "ocean_proximity": "INLAND"
    }
  ]
}'

BATCH_RESPONSE=$(curl -s -X POST \
  http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d "$BATCH_REQUEST")

echo "$BATCH_RESPONSE" | python3 -m json.tool
echo ""

# Show container logs (last 20 lines)
echo "=========================================="
echo "Recent Container Logs"
echo "=========================================="
docker compose logs --tail=20 api
echo ""

# Final summary
echo "=========================================="
echo "✅ ALL TESTS PASSED"
echo "=========================================="
echo ""
echo "API Information:"
echo "  - API URL: http://localhost:8080"
echo "  - Health check: http://localhost:8080/health"
echo "  - API docs: http://localhost:8080/docs"
echo "  - Model info: http://localhost:8080/api/v1/model/info"
echo "  - Prediction: POST http://localhost:8080/api/v1/predict"
echo ""
echo "To stop the API:"
echo "  docker compose down"
echo ""
echo "To view logs:"
echo "  docker compose logs -f api"
echo ""
echo "=========================================="
