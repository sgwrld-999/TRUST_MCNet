"""
TRUST_MCNet API Endpoints

Endpoint handlers and utilities for the TRUST_MCNet API server.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

try:
    from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create mock classes for development without FastAPI
    class BaseModel:
        pass
    class Field:
        def __init__(self, *args, **kwargs):
            pass
    class APIRouter:
        pass
    class HTTPException:
        pass
    class BackgroundTasks:
        pass
    def Query(*args, **kwargs):
        pass

from ..storage.trust_storage import TrustStorage


# Mock API models (for use without FastAPI)
class AdaptationConfigRequest(BaseModel):
    """Request model for configuring adaptive threshold parameters."""
    if FASTAPI_AVAILABLE:
        adaptation_enabled: bool = Field(True, description="Enable/disable adaptation")
        target_accuracy: float = Field(0.85, ge=0.0, le=1.0, description="Target accuracy")
        learning_rate: float = Field(0.01, gt=0.0, le=1.0, description="Adaptation learning rate")
        min_threshold: float = Field(0.1, ge=0.0, le=1.0, description="Minimum threshold")
        max_threshold: float = Field(0.9, ge=0.0, le=1.0, description="Maximum threshold")


class QuarantineConfigRequest(BaseModel):
    """Request model for configuring quarantine parameters."""
    if FASTAPI_AVAILABLE:
        quarantine_enabled: bool = Field(True, description="Enable/disable quarantine")
        quarantine_rounds: int = Field(3, ge=1, le=100, description="Rounds in quarantine")
        min_trust_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Min trust for quarantine")


class ExportRequest(BaseModel):
    """Request model for data export."""
    if FASTAPI_AVAILABLE:
        export_format: str = Field("json", description="Export format (json, csv)")
        include_history: bool = Field(True, description="Include historical data")
        start_round: Optional[int] = Field(None, description="Start round for export")
        end_round: Optional[int] = Field(None, description="End round for export")


def create_trust_api_router(storage: TrustStorage, trust_evaluator=None, trust_strategy=None) -> APIRouter:
    """
    Create API router with all TRUST_MCNet endpoints.
    
    Args:
        storage: TrustStorage instance
        trust_evaluator: TrustEvaluator instance (optional)
        trust_strategy: UnifiedTrustStrategy instance (optional)
        
    Returns:
        Configured APIRouter instance
    """
    if not FASTAPI_AVAILABLE:
        logging.warning("FastAPI not available, returning mock router")
        return APIRouter()
        
    router = APIRouter()
    logger = logging.getLogger(__name__)
    
    # Configuration endpoints
    @router.get("/config/adaptation")
    async def get_adaptation_config():
        """Get current adaptive threshold configuration."""
        try:
            config = {
                "adaptation_enabled": False,
                "target_accuracy": 0.85,
                "learning_rate": 0.01,
                "min_threshold": 0.1,
                "max_threshold": 0.9,
                "current_threshold": 0.5
            }
            
            if trust_strategy and hasattr(trust_strategy, 'get_adaptation_config'):
                config.update(trust_strategy.get_adaptation_config())
            elif trust_evaluator:
                config["current_threshold"] = trust_evaluator.threshold
                
            return config
            
        except Exception as e:
            logger.error(f"Failed to get adaptation config: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    @router.post("/config/adaptation")
    async def update_adaptation_config(request: AdaptationConfigRequest):
        """Update adaptive threshold configuration."""
        try:
            config_dict = request.dict() if hasattr(request, 'dict') else {}
            
            if trust_strategy and hasattr(trust_strategy, 'update_adaptation_config'):
                trust_strategy.update_adaptation_config(config_dict)
            elif trust_evaluator:
                # Basic threshold update
                if 'current_threshold' in config_dict:
                    trust_evaluator.threshold = config_dict['current_threshold']
                    
            logger.info(f"Adaptation config updated: {config_dict}")
            return {"message": "Adaptation configuration updated", "config": config_dict}
            
        except Exception as e:
            logger.error(f"Failed to update adaptation config: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    @router.get("/config/quarantine")
    async def get_quarantine_config():
        """Get current quarantine configuration."""
        try:
            config = {
                "quarantine_enabled": True,
                "quarantine_rounds": 3,
                "min_trust_threshold": 0.3
            }
            
            if trust_evaluator and hasattr(trust_evaluator, 'quarantine_state'):
                q_state = trust_evaluator.quarantine_state
                config.update({
                    "quarantine_enabled": True,
                    "quarantine_rounds": getattr(q_state, 'default_quarantine_rounds', 3),
                    "min_trust_threshold": trust_evaluator.threshold
                })
                
            return config
            
        except Exception as e:
            logger.error(f"Failed to get quarantine config: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    @router.post("/config/quarantine")
    async def update_quarantine_config(request: QuarantineConfigRequest):
        """Update quarantine configuration."""
        try:
            config_dict = request.dict() if hasattr(request, 'dict') else {}
            
            if trust_evaluator and hasattr(trust_evaluator, 'quarantine_state'):
                q_state = trust_evaluator.quarantine_state
                if hasattr(q_state, 'update_config'):
                    q_state.update_config(config_dict)
                    
            logger.info(f"Quarantine config updated: {config_dict}")
            return {"message": "Quarantine configuration updated", "config": config_dict}
            
        except Exception as e:
            logger.error(f"Failed to update quarantine config: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Advanced query endpoints
    @router.get("/analytics/trust-trends")
    async def get_trust_trends(
        client_id: Optional[str] = Query(None, description="Specific client ID"),
        rounds: int = Query(20, ge=1, le=1000, description="Number of rounds to analyze")
    ):
        """Get trust trend analysis."""
        try:
            if client_id:
                # Single client trends
                history = storage.load_client_trust_history(client_id, rounds)
                if not history:
                    raise HTTPException(status_code=404, detail=f"No data for client {client_id}")
                    
                trends = {
                    "client_id": client_id,
                    "trust_history": history,
                    "trend": "stable",  # Would calculate actual trend
                    "volatility": 0.0,  # Would calculate actual volatility
                    "average_trust": sum(history) / len(history)
                }
                
                # Calculate simple trend
                if len(history) >= 2:
                    recent_avg = sum(history[-5:]) / min(5, len(history))
                    older_avg = sum(history[:5]) / min(5, len(history))
                    if recent_avg > older_avg + 0.1:
                        trends["trend"] = "improving"
                    elif recent_avg < older_avg - 0.1:
                        trends["trend"] = "declining"
                        
                return trends
                
            else:
                # Overall trends
                all_trust = storage.load_all_clients_current_trust()
                if not all_trust:
                    return {"message": "No trust data available"}
                    
                client_trends = {}
                for cid in list(all_trust.keys())[:10]:  # Limit to avoid large responses
                    history = storage.load_client_trust_history(cid, rounds)
                    if history:
                        client_trends[cid] = {
                            "current_trust": history[-1],
                            "average_trust": sum(history) / len(history),
                            "data_points": len(history)
                        }
                        
                return {
                    "overall_trends": client_trends,
                    "summary": {
                        "total_clients": len(all_trust),
                        "analyzed_clients": len(client_trends),
                        "rounds_analyzed": rounds
                    }
                }
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get trust trends: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    @router.get("/analytics/quarantine-impact")
    async def get_quarantine_impact():
        """Analyze quarantine effectiveness and impact."""
        try:
            # Get all quarantine events
            all_clients = storage.load_all_clients_current_trust()
            quarantine_analysis = {
                "total_quarantines": 0,
                "clients_affected": 0,
                "average_quarantine_duration": 0,
                "effectiveness_score": 0.0,
                "client_details": []
            }
            
            affected_clients = 0
            total_quarantines = 0
            
            for client_id in all_clients.keys():
                quarantine_history = storage.db.get_quarantine_history(client_id)
                if quarantine_history:
                    affected_clients += 1
                    client_quarantines = len([e for e in quarantine_history if e['event_type'] == 'QUARANTINED'])
                    total_quarantines += client_quarantines
                    
                    # Get trust improvement after quarantine
                    trust_history = storage.load_client_trust_history(client_id, 20)
                    improvement = 0.0
                    if len(trust_history) >= 2:
                        improvement = trust_history[-1] - trust_history[0]
                        
                    quarantine_analysis["client_details"].append({
                        "client_id": client_id,
                        "total_quarantines": client_quarantines,
                        "trust_improvement": improvement,
                        "current_trust": trust_history[-1] if trust_history else 0.0
                    })
                    
            quarantine_analysis.update({
                "total_quarantines": total_quarantines,
                "clients_affected": affected_clients,
                "average_quarantine_duration": 3,  # Default, would calculate from data
                "effectiveness_score": 0.75  # Would calculate based on trust improvements
            })
            
            return quarantine_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze quarantine impact: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    @router.post("/export")
    async def export_data(request: ExportRequest):
        """Export trust and quarantine data."""
        try:
            export_format = getattr(request, 'export_format', 'json')
            include_history = getattr(request, 'include_history', True)
            
            # Collect export data
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "format": export_format,
                    "includes_history": include_history
                },
                "trust_data": {},
                "quarantine_data": {}
            }
            
            # Get current trust scores
            all_trust = storage.load_all_clients_current_trust()
            export_data["trust_data"]["current_scores"] = all_trust
            
            if include_history:
                # Add historical data for each client
                export_data["trust_data"]["history"] = {}
                for client_id in all_trust.keys():
                    history = storage.load_client_trust_history(client_id, 50)
                    export_data["trust_data"]["history"][client_id] = history
                    
                    # Add quarantine history
                    q_history = storage.db.get_quarantine_history(client_id)
                    if q_history:
                        export_data["quarantine_data"][client_id] = q_history
                        
            # Format based on requested format
            if export_format.lower() == 'csv':
                # For CSV, return a simplified format
                csv_data = []
                for client_id, trust in all_trust.items():
                    csv_data.append({
                        "client_id": client_id,
                        "current_trust": trust,
                        "has_quarantine_history": client_id in export_data["quarantine_data"]
                    })
                return {"format": "csv", "data": csv_data}
            else:
                return export_data
                
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    @router.delete("/data/reset")
    async def reset_all_data(confirm: bool = Query(False, description="Confirmation required")):
        """Reset all trust and quarantine data (dangerous operation)."""
        if not confirm:
            raise HTTPException(
                status_code=400, 
                detail="This operation requires confirmation. Add ?confirm=true to the request."
            )
            
        try:
            # Reset storage
            storage.db.reset_database()
            logger.warning("All trust data has been reset via API")
            
            return {
                "message": "All trust and quarantine data has been reset",
                "timestamp": datetime.now().isoformat(),
                "warning": "This action is irreversible"
            }
            
        except Exception as e:
            logger.error(f"Failed to reset data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return router


def setup_api_endpoints(app, storage: TrustStorage, trust_evaluator=None, trust_strategy=None):
    """
    Setup all API endpoints for a FastAPI app.
    
    Args:
        app: FastAPI application instance
        storage: TrustStorage instance
        trust_evaluator: TrustEvaluator instance (optional)
        trust_strategy: UnifiedTrustStrategy instance (optional)
    """
    if not FASTAPI_AVAILABLE:
        logging.warning("FastAPI not available, skipping endpoint setup")
        return
        
    # Create and include router
    trust_router = create_trust_api_router(storage, trust_evaluator, trust_strategy)
    app.include_router(trust_router, prefix="/api/v1", tags=["trust-management"])
    
    logging.info("TRUST_MCNet API endpoints configured successfully")
