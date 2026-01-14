# Streamlit Frontend for Housing Price Prediction

Interactive web interface for predicting California housing prices.

## Features

- 🏠 Interactive input form for house features
- 🗺️ Location preview with map visualization
- 📊 Feature comparison with typical California homes
- 💰 Detailed price analysis and context
- 📈 Real-time predictions from FastAPI backend
- 🎨 Modern, responsive UI with Plotly charts

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set API URL (default: http://localhost:8080)
export API_URL=http://localhost:8080

# Run Streamlit
streamlit run app.py
```

Access at: http://localhost:8501

### Docker

```bash
# Build image
docker build -t housing-streamlit:latest .

# Run container
docker run -d \
  --name housing-streamlit \
  -p 8501:8501 \
  -e API_URL=http://host.docker.internal:8080 \
  housing-streamlit:latest
```

### Docker Compose (with API)

```bash
# From project root
docker-compose up -d

# Access:
# - Streamlit: http://localhost:8501
# - API: http://localhost:8080
```

## Configuration

Environment variables:

- `API_URL`: FastAPI backend URL (default: `http://localhost:8080`)

## Screenshots

### Main Interface
- Input form with location and house features
- Interactive map showing house location
- Real-time API status monitoring

### Prediction Results
- Predicted median house value
- Feature comparison radar chart
- Price context metrics
- Detailed analysis

## Tech Stack

- **Streamlit** 1.31.0: Web framework
- **Plotly** 5.18.0: Interactive charts
- **Requests** 2.31.0: HTTP client
- **Pandas** 2.1.3: Data processing

## Architecture

```
User Browser
     ↓
Streamlit Frontend (8501)
     ↓ HTTP POST
FastAPI Backend (8080)
     ↓
ML Model (scikit-learn)
```

## Development

### Project Structure

```
streamlit_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image
├── .streamlit/         # Streamlit config (generated)
└── README.md           # This file
```

### Adding Features

1. **New visualization**: Add function to `app.py`
2. **New endpoint**: Update `API_*_ENDPOINT` constants
3. **Styling**: Modify CSS in `st.markdown()` sections

## Testing

### Manual Testing

```bash
# 1. Start API (separate terminal)
cd api && uvicorn app.main:app --port 8080

# 2. Start Streamlit
streamlit run app.py

# 3. Test predictions in browser
```

### API Health Check

The app automatically checks API health on load:
- ✅ Green: API healthy, model loaded
- ⚠️ Yellow: API healthy, model not loaded
- ❌ Red: API unavailable

## Troubleshooting

### "API is unavailable"
- Check API is running: `curl http://localhost:8080/health`
- Verify `API_URL` environment variable
- Check Docker network if using containers

### "Connection refused"
- Docker: Use `http://host.docker.internal:8080` for API_URL
- Linux Docker: Use `http://172.17.0.1:8080`
- docker-compose: Use service name `http://api:8080`

### Port already in use
```bash
# Find process using port 8501
lsof -i :8501

# Kill process
kill -9 <PID>
```

## Deployment

### Cloud Run (Streamlit)

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/housing-streamlit:latest

# Deploy
gcloud run deploy housing-streamlit \
  --image gcr.io/PROJECT_ID/housing-streamlit:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "API_URL=https://housing-api-xxxxx.run.app"
```

## License

MIT

## Author

Carlos Daniel Jiménez (danieljimenez88m@gmail.com)
