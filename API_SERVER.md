## Enhanced API Server

TRUST-MCNet provides an enhanced REST API server for managing trust and quarantine operations during federated learning experiments. The API server enables monitoring, configuration and analysis of trust scores, thresholds, and quarantine states.

### Running the API Server

```bash
# Run the enhanced API server with default settings
python examples/enhanced_api_server.py

# Run with custom configuration and port
python examples/enhanced_api_server.py --config config/trust.yaml --port 8080
```

### API Endpoints

#### Threshold Management

- `GET /threshold` - Get current threshold configuration
- `POST /threshold` - Update static threshold value
- `POST /threshold/dynamic` - Configure dynamic threshold behavior

#### Quarantine Management

- `GET /quarantine` - List all quarantined clients
- `GET /quarantine/{client_id}` - Get quarantine status for a client
- `POST /quarantine/{client_id}/release` - Manually release a client

#### Trust Metrics

- `GET /trust/stats` - Get overall trust statistics
- `GET /trust/clients` - List all client IDs
- `GET /trust/clients/{client_id}` - Get trust details for a client

#### Analysis

- `GET /analysis/threshold` - Analyze impact of threshold on performance

### Dynamic Trust Threshold

The enhanced API server supports dynamic threshold calculation based on trust score distribution, round number, and performance metrics. This approach automatically adjusts the threshold based on the current state of the federated learning system.

For more details, see [Dynamic Trust Threshold Documentation](docs/dynamic_threshold.md).

### Example API Usage

```bash
# Get current threshold
curl http://localhost:8081/threshold

# Update dynamic threshold config
curl -X POST http://localhost:8081/threshold/dynamic \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "target_trusted_ratio": 0.7,
      "min_trusted_clients": 3,
      "min_threshold": 0.15,
      "max_threshold": 0.85,
      "percentile_weight": 0.3,
      "statistical_weight": 0.5,
      "adaptive_weight": 0.2
    },
    "enable_dynamic_threshold": true,
    "reason": "Adjusting for increased client participation"
  }'

# Get trust statistics
curl http://localhost:8081/trust/stats

# Get quarantine status
curl http://localhost:8081/quarantine
```
