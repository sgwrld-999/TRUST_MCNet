"""
Enhanced TRUST_MCNet API Server

FastAPI-based REST API server for exposing trust management functionality with improved
features including dynamic threshold management, better error handling, and more robust
integration with the trust evaluation system.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
import json
import numpy as np
from datetime import datetime

from ..trust_module.trust_evaluator import TrustEvaluator
from ..storage.trust_storage import TrustStorage
from ..strategies.unified_trust_strategy import UnifiedTrustStrategy


# Pydantic Models for API
class ThresholdUpdateRequest(BaseModel):
    """Request model for updating adaptive threshold."""
    new_threshold: float = Field(..., ge=0.0, le=1.0, description="New trust threshold")
    reason: str = Field("", description="Reason for threshold change")
    apply_to_future_rounds: bool = Field(False, description="Whether to apply this threshold to future rounds")


class DynamicThresholdConfig(BaseModel):
    """Dynamic threshold configuration."""
    target_trusted_ratio: float = Field(0.6, ge=0.0, le=1.0, 
                                      description="Target ratio of trusted clients")
    min_trusted_clients: int = Field(2, ge=1, 
                                   description="Minimum number of trusted clients")
    min_threshold: float = Field(0.1, ge=0.0, le=0.5, 
                               description="Minimum trust threshold")
    max_threshold: float = Field(0.9, ge=0.5, le=1.0, 
                               description="Maximum trust threshold")
    percentile_weight: float = Field(0.4, ge=0.0, le=1.0,
                                   description="Weight for percentile-based threshold")
    statistical_weight: float = Field(0.4, ge=0.0, le=1.0,
                                    description="Weight for statistical threshold")
    adaptive_weight: float = Field(0.2, ge=0.0, le=1.0, 
                                 description="Weight for adaptive threshold")

    @validator('percentile_weight', 'statistical_weight', 'adaptive_weight')
    def weights_sum_to_one(cls, v, values):
        """Validate that weights sum to approximately 1."""
        if 'percentile_weight' in values and 'statistical_weight' in values:
            total = values['percentile_weight'] + values['statistical_weight'] + v
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                raise ValueError(f"Weights must sum to 1.0 (current sum: {total})")
        return v


class ThresholdResponse(BaseModel):
    """Response model for threshold operations."""
    current_threshold: float
    target_accuracy: float
    adaptation_enabled: bool
    last_updated_round: Optional[int] = None
    dynamic_threshold_enabled: bool = False
    dynamic_config: Optional[DynamicThresholdConfig] = None


class DynamicThresholdUpdateRequest(BaseModel):
    """Request model for updating dynamic threshold configuration."""
    config: DynamicThresholdConfig
    enable_dynamic_threshold: bool = Field(True, description="Whether to enable dynamic threshold")
    reason: str = Field("", description="Reason for configuration change")


class QuarantineStatusResponse(BaseModel):
    """Response model for quarantine status."""
    client_id: str
    is_quarantined: bool
    quarantine_rounds_left: int
    total_quarantines: int
    last_quarantine_reason: Optional[str] = None


class TrustStatsResponse(BaseModel):
    """Response model for trust statistics."""
    total_clients: int
    mean_trust: float
    min_trust: float
    max_trust: float
    clients_quarantined: int
    latest_round: int
    trust_score_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of trust scores in bins (0.0-0.1, 0.1-0.2, etc.)"
    )
    threshold_history: List[float] = Field(
        default_factory=list,
        description="History of threshold values"
    )


class ClientTrustResponse(BaseModel):
    """Response model for client trust information."""
    client_id: str
    current_trust: float
    trust_history: List[float]
    performance_metrics: Dict[str, Any]
    quarantine_status: QuarantineStatusResponse
    trust_components: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual trust components (cosine, entropy, etc.)"
    )


class EnhancedTrustMCNetAPIServer:
    """
    Enhanced REST API server for TRUST_MCNet management.
    
    Provides endpoints for:
    - Adaptive threshold control including dynamic threshold management
    - Quarantine management with detailed status tracking
    - Trust metrics monitoring with component-level details
    - Real-time status updates and historical analysis
    """
    
    def __init__(
        self,
        trust_evaluator: Optional[TrustEvaluator] = None,
        trust_strategy: Optional[UnifiedTrustStrategy] = None,
        storage: Optional[TrustStorage] = None,
        host: str = "0.0.0.0",
        port: int = 8081
    ):
        """
        Initialize enhanced API server.
        
        Args:
            trust_evaluator: TrustEvaluator instance
            trust_strategy: UnifiedTrustStrategy instance
            storage: TrustStorage instance
            host: Server host
            port: Server port
        """
        self.trust_evaluator = trust_evaluator or TrustEvaluator()
        self.trust_strategy = trust_strategy
        self.storage = storage or TrustStorage()
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        
        # Create FastAPI app
        self.app = self._create_app()
        
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            self.logger.info("Enhanced TRUST_MCNet API Server starting up...")
            yield
            # Shutdown
            self.logger.info("Enhanced TRUST_MCNet API Server shutting down...")
        
        app = FastAPI(
            title="TRUST_MCNet Enhanced API",
            description="REST API for TRUST_MCNet federated learning trust management with dynamic threshold support",
            version="2.0.0",
            lifespan=lifespan
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routes
        self._add_routes(app)
        
        return app
        
    def _add_routes(self, app: FastAPI) -> None:
        """Add API routes to FastAPI app."""
        
        @app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint with API information."""
            return {
                "name": "TRUST_MCNet Enhanced API",
                "version": "2.0.0",
                "status": "running",
                "description": "REST API for federated learning trust management with dynamic threshold support"
            }
            
        @app.get("/health", response_model=Dict[str, str])
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": str(datetime.now())}
            
        # Adaptive threshold endpoints
        @app.get("/threshold", response_model=ThresholdResponse)
        async def get_threshold():
            """Get current adaptive threshold information."""
            try:
                # Check if dynamic threshold is enabled in the trust evaluator
                dynamic_enabled = hasattr(self.trust_evaluator, '_dynamic_threshold_initialized') and \
                                getattr(self.trust_evaluator, '_dynamic_threshold_initialized', False)
                
                if self.trust_strategy and hasattr(self.trust_strategy, 'get_adaptation_status'):
                    status = self.trust_strategy.get_adaptation_status()
                    
                    # Get dynamic threshold config if available
                    dynamic_config = None
                    if dynamic_enabled:
                        dynamic_config = DynamicThresholdConfig(
                            target_trusted_ratio=getattr(self.trust_evaluator, 'target_trusted_ratio', 0.6),
                            min_trusted_clients=getattr(self.trust_evaluator, 'min_trusted_clients', 2),
                            min_threshold=getattr(self.trust_evaluator, 'min_threshold', 0.1),
                            max_threshold=getattr(self.trust_evaluator, 'max_threshold', 0.9),
                            percentile_weight=getattr(self.trust_evaluator, 'percentile_weight', 0.4),
                            statistical_weight=getattr(self.trust_evaluator, 'statistical_weight', 0.4),
                            adaptive_weight=getattr(self.trust_evaluator, 'adaptive_weight', 0.2),
                        )
                    
                    return ThresholdResponse(
                        current_threshold=status.get('current_trust_threshold', 0.5),
                        target_accuracy=status.get('target_accuracy', 0.85),
                        adaptation_enabled=status.get('adaptation_enabled', False),
                        last_updated_round=status.get('last_adaptation_round'),
                        dynamic_threshold_enabled=dynamic_enabled,
                        dynamic_config=dynamic_config
                    )
                else:
                    # Fall back to basic trust evaluator
                    dynamic_config = None
                    if dynamic_enabled:
                        dynamic_config = DynamicThresholdConfig(
                            target_trusted_ratio=getattr(self.trust_evaluator, 'target_trusted_ratio', 0.6),
                            min_trusted_clients=getattr(self.trust_evaluator, 'min_trusted_clients', 2),
                            min_threshold=getattr(self.trust_evaluator, 'min_threshold', 0.1),
                            max_threshold=getattr(self.trust_evaluator, 'max_threshold', 0.9),
                            percentile_weight=getattr(self.trust_evaluator, 'percentile_weight', 0.4),
                            statistical_weight=getattr(self.trust_evaluator, 'statistical_weight', 0.4),
                            adaptive_weight=getattr(self.trust_evaluator, 'adaptive_weight', 0.2),
                        )
                    
                    return ThresholdResponse(
                        current_threshold=self.trust_evaluator.threshold,
                        target_accuracy=0.85,
                        adaptation_enabled=False,
                        dynamic_threshold_enabled=dynamic_enabled,
                        dynamic_config=dynamic_config
                    )
            except Exception as e:
                self.logger.error(f"Failed to get threshold: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @app.post("/threshold", response_model=ThresholdResponse)
        async def update_threshold(request: ThresholdUpdateRequest):
            """Update adaptive threshold value."""
            try:
                # Update threshold in trust evaluator
                self.trust_evaluator.threshold = request.new_threshold
                
                # Update in strategy if available
                if self.trust_strategy:
                    self.trust_strategy.trust_threshold = request.new_threshold
                    
                # Record change in storage
                if hasattr(self.trust_strategy, 'round_counter'):
                    round_num = getattr(self.trust_strategy, 'round_counter', 0)
                    self.storage.record_threshold_change(
                        round_number=round_num,
                        new_threshold=request.new_threshold,
                        target_accuracy=getattr(self.trust_strategy, 'target_accuracy', 0.85),
                        current_accuracy=0.0,  # Would need actual current accuracy
                        reason=request.reason
                    )
                    
                self.logger.info(f"Threshold updated to {request.new_threshold}: {request.reason}")
                
                return await get_threshold()
                
            except Exception as e:
                self.logger.error(f"Failed to update threshold: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/threshold/dynamic", response_model=ThresholdResponse)
        async def update_dynamic_threshold_config(request: DynamicThresholdUpdateRequest):
            """Update dynamic threshold configuration."""
            try:
                # Ensure dynamic threshold system is initialized
                if not hasattr(self.trust_evaluator, '_dynamic_threshold_initialized'):
                    if not hasattr(self.trust_evaluator, '__init_dynamic_threshold_system'):
                        raise HTTPException(
                            status_code=400, 
                            detail="Dynamic threshold not supported by this trust evaluator"
                        )
                    self.trust_evaluator.__init_dynamic_threshold_system()
                
                # Update configuration
                self.trust_evaluator.target_trusted_ratio = request.config.target_trusted_ratio
                self.trust_evaluator.min_trusted_clients = request.config.min_trusted_clients
                self.trust_evaluator.min_threshold = request.config.min_threshold
                self.trust_evaluator.max_threshold = request.config.max_threshold
                self.trust_evaluator.percentile_weight = request.config.percentile_weight
                self.trust_evaluator.statistical_weight = request.config.statistical_weight
                self.trust_evaluator.adaptive_weight = request.config.adaptive_weight
                
                # Enable/disable dynamic threshold
                self.trust_evaluator._dynamic_threshold_initialized = request.enable_dynamic_threshold
                
                # Log change
                self.logger.info(
                    f"Dynamic threshold {'enabled' if request.enable_dynamic_threshold else 'disabled'} "
                    f"with configuration: {request.config.dict()}"
                )
                
                # Record change
                if hasattr(self.trust_strategy, 'round_counter'):
                    round_num = getattr(self.trust_strategy, 'round_counter', 0)
                    self.storage.record_threshold_change(
                        round_number=round_num,
                        new_threshold=self.trust_evaluator.threshold,
                        target_accuracy=getattr(self.trust_strategy, 'target_accuracy', 0.85),
                        current_accuracy=0.0,
                        reason=f"Dynamic threshold configuration update: {request.reason}"
                    )
                
                return await get_threshold()
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to update dynamic threshold config: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        # Quarantine endpoints
        @app.get("/quarantine", response_model=List[QuarantineStatusResponse])
        async def get_all_quarantine_status():
            """Get quarantine status for all clients."""
            try:
                if not hasattr(self.trust_evaluator, 'quarantine_state'):
                    return []
                    
                quarantine_state = self.trust_evaluator.quarantine_state
                all_status = []
                
                # Get all clients that have been tracked
                for client_id, client_state in quarantine_state.client_states.items():
                    quarantine_history = self.storage.db.get_quarantine_history(client_id)
                    
                    last_reason = None
                    if quarantine_history:
                        last_quarantine = next(
                            (event for event in reversed(quarantine_history) 
                             if event['event_type'] == 'QUARANTINED'), None
                        )
                        if last_quarantine:
                            last_reason = last_quarantine['reason']
                    
                    all_status.append(QuarantineStatusResponse(
                        client_id=client_id,
                        is_quarantined=client_state.is_quarantined,
                        quarantine_rounds_left=client_state.quarantine_rounds,
                        total_quarantines=getattr(client_state, 'total_quarantines', 0),
                        last_quarantine_reason=last_reason
                    ))
                    
                return all_status
                
            except Exception as e:
                self.logger.error(f"Failed to get quarantine status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @app.get("/quarantine/{client_id}", response_model=QuarantineStatusResponse)
        async def get_quarantine_status(client_id: str):
            """Get quarantine status for a specific client."""
            try:
                if not hasattr(self.trust_evaluator, 'quarantine_state'):
                    raise HTTPException(status_code=404, detail="Quarantine not enabled")
                    
                quarantine_state = self.trust_evaluator.quarantine_state
                
                if client_id not in quarantine_state.client_states:
                    raise HTTPException(status_code=404, detail=f"Client {client_id} not found")
                
                client_state = quarantine_state.client_states[client_id]
                quarantine_history = self.storage.db.get_quarantine_history(client_id)
                
                last_reason = None
                if quarantine_history:
                    last_quarantine = next(
                        (event for event in reversed(quarantine_history) 
                         if event['event_type'] == 'QUARANTINED'), None
                    )
                    if last_quarantine:
                        last_reason = last_quarantine['reason']
                
                return QuarantineStatusResponse(
                    client_id=client_id,
                    is_quarantined=client_state.is_quarantined,
                    quarantine_rounds_left=client_state.quarantine_rounds,
                    total_quarantines=getattr(client_state, 'total_quarantines', 0),
                    last_quarantine_reason=last_reason
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get quarantine status for {client_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @app.post("/quarantine/{client_id}/release")
        async def release_from_quarantine(client_id: str, background_tasks: BackgroundTasks):
            """Manually release a client from quarantine."""
            try:
                if not hasattr(self.trust_evaluator, 'quarantine_state'):
                    raise HTTPException(status_code=404, detail="Quarantine not enabled")
                    
                quarantine_state = self.trust_evaluator.quarantine_state
                
                if client_id not in quarantine_state.client_states:
                    raise HTTPException(status_code=404, detail=f"Client {client_id} not found")
                
                client_state = quarantine_state.client_states[client_id]
                
                if not client_state.is_quarantined:
                    raise HTTPException(status_code=400, detail="Client is not quarantined")
                    
                # Reset quarantine status
                client_state.is_quarantined = False
                client_state.quarantine_rounds = 0
                
                # Record release event
                background_tasks.add_task(
                    self.storage.record_quarantine,
                    client_id=client_id,
                    round_number=getattr(self.trust_strategy, 'round_counter', 0),
                    is_quarantined=False,
                    reason="Manual release via API"
                )
                
                self.logger.info(f"Client {client_id} manually released from quarantine")
                
                return {"message": f"Client {client_id} released from quarantine"}
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to release {client_id} from quarantine: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        # Trust metrics endpoints
        @app.get("/trust/stats", response_model=TrustStatsResponse)
        async def get_trust_statistics():
            """Get overall trust statistics."""
            try:
                stats = self.storage.get_storage_stats()
                
                # Enhanced statistics
                trust_scores = stats.get('all_trust_scores', [])
                threshold_history = []
                
                # Create trust score distribution
                dist = {}
                if trust_scores:
                    for score in trust_scores:
                        bin_key = f"{int(score * 10) / 10:.1f}-{int(score * 10 + 1) / 10:.1f}"
                        dist[bin_key] = dist.get(bin_key, 0) + 1
                
                # Get threshold history if available
                if hasattr(self.trust_evaluator, 'threshold_history'):
                    threshold_history = self.trust_evaluator.threshold_history
                
                return TrustStatsResponse(
                    total_clients=stats.get('total_clients', 0),
                    mean_trust=stats.get('mean_trust', 0.0),
                    min_trust=stats.get('min_trust', 0.0),
                    max_trust=stats.get('max_trust', 0.0),
                    clients_quarantined=stats.get('clients_quarantined', 0),
                    latest_round=stats.get('latest_round', 0),
                    trust_score_distribution=dist,
                    threshold_history=threshold_history
                )
                
            except Exception as e:
                self.logger.error(f"Failed to get trust statistics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @app.get("/trust/clients", response_model=List[str])
        async def get_all_client_ids():
            """Get list of all tracked client IDs."""
            try:
                all_trust = self.storage.load_all_clients_current_trust()
                return list(all_trust.keys())
            except Exception as e:
                self.logger.error(f"Failed to get client IDs: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @app.get("/trust/clients/{client_id}", response_model=ClientTrustResponse)
        async def get_client_trust_info(client_id: str, history_rounds: int = Query(10, ge=1, le=100)):
            """Get detailed trust information for a specific client."""
            try:
                # Get trust history
                trust_history = self.storage.load_client_trust_history(client_id, history_rounds)
                current_trust = trust_history[-1] if trust_history else 0.0
                
                # Get quarantine status
                try:
                    quarantine_status = await get_quarantine_status(client_id)
                except HTTPException:
                    # Create default status if client not found in quarantine system
                    quarantine_status = QuarantineStatusResponse(
                        client_id=client_id,
                        is_quarantined=False,
                        quarantine_rounds_left=0,
                        total_quarantines=0
                    )
                
                # Get performance metrics (latest)
                performance_data = self.storage.db.get_trust_history(client_id, 1)
                performance_metrics = {}
                if performance_data:
                    latest = performance_data[0]
                    performance_metrics = {
                        'accuracy': latest.get('accuracy', 0.0),
                        'loss': latest.get('loss', 1.0),
                        'round': latest.get('round_number', 0)
                    }
                
                # Get trust components
                trust_components = {}
                raw_data = self.storage.db.get_trust_history(client_id, 1)
                if raw_data:
                    latest = raw_data[0]
                    if 'raw_data' in latest and latest['raw_data']:
                        try:
                            raw_json = json.loads(latest['raw_data'])
                            if isinstance(raw_json, dict):
                                for comp in ['cosine_score', 'entropy_score', 'reputation_score']:
                                    if comp in raw_json:
                                        trust_components[comp.split('_')[0]] = raw_json[comp]
                        except Exception as e:
                            self.logger.warning(f"Failed to parse trust components: {e}")
                
                return ClientTrustResponse(
                    client_id=client_id,
                    current_trust=current_trust,
                    trust_history=trust_history,
                    performance_metrics=performance_metrics,
                    quarantine_status=quarantine_status,
                    trust_components=trust_components
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get trust info for {client_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Dynamic threshold analysis endpoints
        @app.get("/analysis/threshold")
        async def analyze_threshold_impact(rounds: int = Query(10, ge=1, le=100)):
            """Analyze impact of threshold changes on system performance."""
            try:
                # Get threshold history
                threshold_history = []
                if hasattr(self.trust_evaluator, 'threshold_history'):
                    threshold_history = self.trust_evaluator.threshold_history[-rounds:]
                
                # Get performance history
                performance_history = []
                if hasattr(self.trust_evaluator, 'performance_history'):
                    performance_history = self.trust_evaluator.performance_history[-rounds:]
                
                # Get participation data
                participation_data = []
                if self.trust_strategy and hasattr(self.trust_strategy, 'participation_history'):
                    participation_data = self.trust_strategy.participation_history[-rounds:]
                
                # Analyze correlation between threshold and performance
                correlation = None
                if len(threshold_history) > 2 and len(performance_history) > 2:
                    if len(threshold_history) == len(performance_history):
                        try:
                            correlation = np.corrcoef(threshold_history, performance_history)[0, 1]
                        except Exception:
                            pass
                
                return {
                    "threshold_history": threshold_history,
                    "performance_history": performance_history,
                    "participation_data": participation_data,
                    "threshold_performance_correlation": correlation,
                    "analysis_summary": self._generate_threshold_analysis(
                        threshold_history, performance_history, participation_data
                    )
                }
            except Exception as e:
                self.logger.error(f"Failed to analyze threshold impact: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _generate_threshold_analysis(self, thresholds, performances, participation):
        """Generate analysis summary of threshold impact."""
        if not thresholds or not performances:
            return "Insufficient data for analysis"
        
        summary = []
        
        # Check trends
        threshold_trend = "stable"
        if len(thresholds) > 2:
            if all(a < b for a, b in zip(thresholds[:-1], thresholds[1:])):
                threshold_trend = "increasing"
            elif all(a > b for a, b in zip(thresholds[:-1], thresholds[1:])):
                threshold_trend = "decreasing"
            elif sum(1 for a, b in zip(thresholds[:-1], thresholds[1:]) if a < b) > len(thresholds) / 2:
                threshold_trend = "mostly increasing"
            elif sum(1 for a, b in zip(thresholds[:-1], thresholds[1:]) if a > b) > len(thresholds) / 2:
                threshold_trend = "mostly decreasing"
        
        summary.append(f"Threshold trend is {threshold_trend}")
        
        # Performance correlation
        if len(thresholds) == len(performances) and len(thresholds) > 2:
            try:
                corr = np.corrcoef(thresholds, performances)[0, 1]
                if abs(corr) < 0.2:
                    summary.append("No significant correlation between threshold and performance")
                elif corr > 0:
                    summary.append(f"Positive correlation ({corr:.2f}) between threshold and performance")
                else:
                    summary.append(f"Negative correlation ({corr:.2f}) between threshold and performance")
            except Exception:
                pass
        
        # Participation impact
        if participation and len(participation) > 0:
            avg_participation = sum(p.get('total_participants', 0) for p in participation) / len(participation)
            summary.append(f"Average participation rate: {avg_participation:.1f} clients per round")
            
            if len(participation) > 1:
                trend = sum(1 for i in range(len(participation)-1) 
                           if participation[i+1].get('total_participants', 0) > 
                              participation[i].get('total_participants', 0))
                if trend > len(participation) / 2:
                    summary.append("Participation is trending upward")
                elif trend < len(participation) / 2:
                    summary.append("Participation is trending downward")
                else:
                    summary.append("Participation is stable")
        
        return ". ".join(summary)

    async def start_server(self):
        """Start the API server."""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    def run_server(self):
        """Run the API server (blocking)."""
        uvicorn.run(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
