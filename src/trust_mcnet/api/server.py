"""
TRUST_MCNet API Server

FastAPI-based REST API server for exposing trust management functionality.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ..trust_module.trust_evaluator import TrustEvaluator
from ..storage.trust_storage import TrustStorage
from ..strategies.unified_trust_strategy import UnifiedTrustStrategy


# Pydantic Models for API
class ThresholdUpdateRequest(BaseModel):
    """Request model for updating adaptive threshold."""
    new_threshold: float = Field(..., ge=0.0, le=1.0, description="New trust threshold")
    reason: str = Field("", description="Reason for threshold change")


class ThresholdResponse(BaseModel):
    """Response model for threshold operations."""
    current_threshold: float
    target_accuracy: float
    adaptation_enabled: bool
    last_updated_round: Optional[int] = None


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


class ClientTrustResponse(BaseModel):
    """Response model for client trust information."""
    client_id: str
    current_trust: float
    trust_history: List[float]
    performance_metrics: Dict[str, Any]
    quarantine_status: QuarantineStatusResponse


class TrustMCNetAPIServer:
    """
    REST API server for TRUST_MCNet management.
    
    Provides endpoints for:
    - Adaptive threshold control
    - Quarantine management
    - Trust metrics monitoring
    - Real-time status updates
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
        Initialize API server.
        
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
            self.logger.info("TRUST_MCNet API Server starting up...")
            yield
            # Shutdown
            self.logger.info("TRUST_MCNet API Server shutting down...")
        
        app = FastAPI(
            title="TRUST_MCNet API",
            description="REST API for TRUST_MCNet federated learning trust management",
            version="1.0.0",
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
                "name": "TRUST_MCNet API",
                "version": "1.0.0",
                "status": "running",
                "description": "REST API for federated learning trust management"
            }
            
        @app.get("/health", response_model=Dict[str, str])
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": str(asyncio.get_event_loop().time())}
            
        # Adaptive threshold endpoints
        @app.get("/threshold", response_model=ThresholdResponse)
        async def get_threshold():
            """Get current adaptive threshold information."""
            try:
                if self.trust_strategy and hasattr(self.trust_strategy, 'get_adaptation_status'):
                    status = self.trust_strategy.get_adaptation_status()
                    return ThresholdResponse(
                        current_threshold=status.get('current_trust_threshold', 0.5),
                        target_accuracy=status.get('target_accuracy', 0.85),
                        adaptation_enabled=status.get('adaptation_enabled', False),
                        last_updated_round=status.get('last_adaptation_round')
                    )
                else:
                    return ThresholdResponse(
                        current_threshold=self.trust_evaluator.threshold,
                        target_accuracy=0.85,
                        adaptation_enabled=False
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
                for client_id in quarantine_state._client_status.keys():
                    status = quarantine_state.get_client_status(client_id)
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
                        is_quarantined=quarantine_state.is_quarantined(client_id),
                        quarantine_rounds_left=status.quarantine_rounds_left,
                        total_quarantines=status.total_quarantines,
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
                status = quarantine_state.get_client_status(client_id)
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
                    is_quarantined=quarantine_state.is_quarantined(client_id),
                    quarantine_rounds_left=status.quarantine_rounds_left,
                    total_quarantines=status.total_quarantines,
                    last_quarantine_reason=last_reason
                )
                
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
                
                if not quarantine_state.is_quarantined(client_id):
                    raise HTTPException(status_code=400, detail="Client is not quarantined")
                    
                # Reset quarantine status
                status = quarantine_state.get_client_status(client_id)
                status.quarantine_rounds_left = 0
                
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
                
                return TrustStatsResponse(
                    total_clients=stats.get('total_clients', 0),
                    mean_trust=stats.get('mean_trust', 0.0),
                    min_trust=stats.get('min_trust', 0.0),
                    max_trust=stats.get('max_trust', 0.0),
                    clients_quarantined=stats.get('clients_quarantined', 0),
                    latest_round=stats.get('latest_round', 0)
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
        async def get_client_trust_info(client_id: str, history_rounds: int = 10):
            """Get detailed trust information for a specific client."""
            try:
                # Get trust history
                trust_history = self.storage.load_client_trust_history(client_id, history_rounds)
                current_trust = trust_history[-1] if trust_history else 0.0
                
                # Get quarantine status
                quarantine_status = await get_quarantine_status(client_id)
                
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
                
                return ClientTrustResponse(
                    client_id=client_id,
                    current_trust=current_trust,
                    trust_history=trust_history,
                    performance_metrics=performance_metrics,
                    quarantine_status=quarantine_status
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get trust info for {client_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
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
