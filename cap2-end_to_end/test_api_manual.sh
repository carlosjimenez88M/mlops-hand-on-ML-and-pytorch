#!/bin/bash
# =================================================================
# Manual API Test Script (requires API to be already running)
# Purpose: Quick manual tests for the Housing Price Prediction API
# Usage: ./test_api_manual.sh
# =================================================================

echo "=========================================="
echo "Manual API Tests"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test 1: Health check
echo -e "\n${YELLOW}Test 1: Health Check${NC}"
echo "Request: GET http://localhost:8080/health"
echo "Response:"
curl -s http://localhost:8080/health | python3 -m json.tool
echo ""

# Test 2: Root endpoint
echo -e "\n${YELLOW}Test 2: Root Endpoint${NC}"
echo "Request: GET http://localhost:8080/"
echo "Response:"
curl -s http://localhost:8080/ | python3 -m json.tool
echo ""

# Test 3: Model info
echo -e "\n${YELLOW}Test 3: Model Info${NC}"
echo "Request: GET http://localhost:8080/api/v1/model/info"
echo "Response:"
curl -s http://localhost:8080/api/v1/model/info | python3 -m json.tool
echo ""

# Test 4: Single prediction
echo -e "\n${YELLOW}Test 4: Single Prediction${NC}"
echo "Request: POST http://localhost:8080/api/v1/predict"
echo "Input: San Francisco Bay Area house"
echo "Response:"
curl -s -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }' | python3 -m json.tool
echo ""

# Test 5: Batch prediction
echo -e "\n${YELLOW}Test 5: Batch Prediction (3 houses)${NC}"
echo "Request: POST http://localhost:8080/api/v1/predict"
echo "Input: 3 different California locations"
echo "Response:"
curl -s -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }' | python3 -m json.tool
echo ""

echo "=========================================="
echo -e "${GREEN}✅ All manual tests completed${NC}"
echo "=========================================="
echo ""
echo "API Endpoints:"
echo "  - Health: http://localhost:8080/health"
echo "  - Docs: http://localhost:8080/docs"
echo "  - Model Info: http://localhost:8080/api/v1/model/info"
echo "  - Predict: POST http://localhost:8080/api/v1/predict"
echo ""
