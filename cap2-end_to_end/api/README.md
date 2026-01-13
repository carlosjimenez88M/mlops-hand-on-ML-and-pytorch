# Housing Price Prediction API

FastAPI service for predicting California housing prices using trained ML models.

## Features

- RESTful API with automatic OpenAPI documentation
- Pydantic validation for request/response
- Model loading from GCS or local filesystem
- Health check endpoint
- Production-ready with Cloud Run deployment
- Comprehensive test suite

## Quick Start

### Local Development

1. **Install dependencies**:
```bash
cd api
pip install -r requirements.txt
```

2. **Run the API**:
```bash
uvicorn app.main:app --reload --port 8080
```

3. **Access the API**:
   - API: http://localhost:8080
   - Interactive docs: http://localhost:8080/docs
   - Health check: http://localhost:8080/health

### Using Docker Compose

```bash
# From project root
make compose-up

# Stop services
make compose-down
```

### Using Makefile (from project root)

```bash
# Install API dependencies
make api-install

# Run locally
make api-local

# Run tests
make api-test

# Run tests with coverage
make api-test-cov

# Build Docker image
make api-build
```

## API Endpoints

### Root Endpoint

**GET** `/`

Returns basic API information.

**Response**:
```json
{
  "name": "Housing Price Prediction API",
  "version": "1.0.0",
  "description": "Housing Price Prediction API",
  "docs_url": "/docs",
  "health_url": "/health"
}
```

### Health Check

**GET** `/health`

Check API health and model status.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Predict

**POST** `/api/v1/predict`

Make predictions for housing prices.

**Request Body**:
```json
{
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
}
```

**Response**:
```json
{
  "predictions": [
    {
      "predicted_price": 452600.0,
      "confidence_interval": null
    }
  ],
  "model_version": "randomforest_v1"
}
```

**Validation Rules**:
- `longitude`: -180 to 180
- `latitude`: -90 to 90
- All numeric fields must be >= 0
- `ocean_proximity`: Must be one of:
  - `<1H OCEAN`
  - `INLAND`
  - `ISLAND`
  - `NEAR BAY`
  - `NEAR OCEAN`

### Model Info

**GET** `/api/v1/model/info`

Get information about the loaded model.

**Response**:
```json
{
  "status": "ready",
  "model_loaded": true,
  "model_version": "randomforest_best"
}
```

## Configuration

Configuration is managed through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | API port | 8080 |
| `GCS_BUCKET` | GCS bucket for model storage | "" |
| `GCS_MODEL_PATH` | Path to model in GCS | models/05-selection/randomforest_best.pkl |
| `MODEL_PATH` | Local model path | models/best_model.pkl |
| `WANDB_API_KEY` | W&B API key for logging | "" |
| `WANDB_PROJECT` | W&B project name | housing-mlops-api |

Create a `.env` file in the project root:

```bash
GCS_BUCKET_NAME=your-bucket-name
GCP_PROJECT_ID=your-project-id
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=housing-mlops-api
```

## Weights & Biases Monitoring

The API automatically logs predictions and performance metrics to Weights & Biases for production monitoring.

### What Gets Logged

1. **Prediction Metrics**:
   - Number of predictions per request
   - Mean, min, max predicted values
   - Response time (milliseconds)
   - Model version used

2. **Feature Distributions**:
   - Input feature values (sampled for batch requests)
   - Helps detect data drift

3. **Errors**:
   - Validation errors
   - Prediction failures
   - Error counts and types

4. **Health Checks**:
   - API status
   - Model loaded status

### Enable W&B Logging

Set the `WANDB_API_KEY` environment variable:

```bash
export WANDB_API_KEY=your-key-here
```

The API will automatically initialize W&B logging on startup and log all predictions.

### View Metrics

Go to your W&B dashboard:
```
https://wandb.ai/your-username/housing-mlops-api
```

### Disable W&B Logging

To disable W&B logging:
```bash
unset WANDB_API_KEY
```

Or set it to an empty string in your `.env` file.

### Production Monitoring Use Cases

1. **Data Drift Detection**: Monitor input feature distributions
2. **Performance Tracking**: Track response times and throughput
3. **Error Analysis**: Identify common failure patterns
4. **Model Performance**: Compare predictions across different model versions
5. **Usage Analytics**: Understand API usage patterns

## Testing

### Run Tests

```bash
cd api
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing
```

### Test Coverage

The test suite includes:
- Unit tests for all endpoints
- Input validation tests
- Error handling tests
- Model loading tests
- Health check tests

## Deployment to Cloud Run

### Prerequisites

1. Install Google Cloud SDK
2. Authenticate: `gcloud auth login`
3. Set project: `gcloud config set project YOUR_PROJECT_ID`

### Quick Deploy

```bash
cd api
export GCP_PROJECT_ID=your-project-id
export GCS_BUCKET_NAME=your-bucket-name
./deploy_to_cloudrun.sh
```

### Manual Deployment

1. **Build container image**:
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/housing-price-api
```

2. **Deploy to Cloud Run**:
```bash
gcloud run deploy housing-price-api \
  --image gcr.io/PROJECT_ID/housing-price-api \
  --platform managed \
  --region us-central1 \
  --port 8080 \
  --memory 2Gi \
  --allow-unauthenticated \
  --set-env-vars "GCS_BUCKET_NAME=your-bucket,GCP_PROJECT_ID=your-project"
```

### Using YAML Configuration

```bash
gcloud run services replace cloudrun.yaml
```

## Production Recommendations

### Security

1. **Authentication**: Enable Cloud Run authentication for production:
```bash
gcloud run deploy housing-price-api --no-allow-unauthenticated
```

2. **CORS**: Update CORS settings in `app/main.py` for production:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

3. **API Keys**: Implement API key validation for production use

### Performance

1. **Min Instances**: Set minimum instances for reduced cold starts:
```bash
gcloud run services update housing-price-api --min-instances=1
```

2. **Caching**: Consider implementing response caching for repeated requests

3. **Model Optimization**: Use model quantization or ONNX for faster inference

### Monitoring

1. **Cloud Logging**: All logs are automatically captured in Cloud Logging

2. **Cloud Monitoring**: Set up alerts for:
   - High error rates
   - Slow response times
   - Low model accuracy

3. **Custom Metrics**: Log prediction metrics to track model performance

### Cost Optimization

1. Use CPU allocation only during request processing (default)
2. Set appropriate max instances based on expected traffic
3. Use min-instances=0 for low-traffic applications
4. Monitor and optimize memory/CPU allocation

## Architecture

```
api/
 app/
    __init__.py
    main.py                 # FastAPI application
    core/
       config.py           # Configuration
       model_loader.py     # Model loading logic
    models/
       schemas.py          # Pydantic models
    routers/
        predict.py          # Prediction endpoints
 tests/
    conftest.py            # Test fixtures
    test_api.py            # API tests
 Dockerfile                 # Container definition
 requirements.txt           # Python dependencies
 pytest.ini                 # Pytest configuration
 deploy_to_cloudrun.sh     # Deployment script
 cloudrun.yaml             # Cloud Run config
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid input)
- `422`: Validation error
- `500`: Internal server error

Error responses include detailed messages:

```json
{
  "error": "Invalid input data",
  "detail": "ocean_proximity must be one of: <1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN"
}
```

## Model Loading Strategy

The API attempts to load models in this order:

1. **Google Cloud Storage** (if `GCS_BUCKET` is configured)
2. **Local filesystem** (fallback to `MODEL_PATH`)

Models are loaded once at startup and cached for subsequent requests.

## Troubleshooting

### Model not loading

**Error**: `Model file not found`

**Solution**: Ensure the model exists at the configured path:
- For local: Check `MODEL_PATH` points to valid pickle file
- For GCS: Verify `GCS_BUCKET` and `GCS_MODEL_PATH` are correct

### Import errors

**Error**: `Import "app.X" could not be resolved`

**Solution**: Ensure you're running from the `api` directory or have proper PYTHONPATH:
```bash
export PYTHONPATH=/path/to/api:$PYTHONPATH
```

### Port already in use

**Error**: `Address already in use`

**Solution**: Change port or kill existing process:
```bash
# Use different port
uvicorn app.main:app --port 8081

# Or kill existing process
lsof -ti:8080 | xargs kill
```

## Contributing

1. Add new features in appropriate modules
2. Write tests for all new endpoints
3. Update this README with new endpoints/features
4. Ensure all tests pass before committing

## License

This project is part of the MLOps Housing Price Prediction pipeline.

## Contact

For questions or issues, please contact: danieljimenez88m@gmail.com
