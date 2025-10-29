"""
FastAPI server to expose diagnostic data endpoint for dual-channel implementation.
This allows the frontend to retrieve diagnostic data separately from TTS output.
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tools.RAG_tools import get_latest_diagnostic_data, clear_diagnostic_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Allion Diagnostic API",
    description="API endpoints for retrieving diagnostic data in dual-channel implementation",
    version="1.0.0"
)

# Configure CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",              # Local development
        "https://frontendtest5.vercel.app",   # Production Vercel app
        "https://*.vercel.app",               # All Vercel domains (for branch previews)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Allion Diagnostic API is running", "status": "healthy"}

@app.get("/api/diagnostic-data")
async def get_diagnostic_data():
    """
    Retrieve the latest diagnostic data stored by the backend.
    
    Returns:
        - diagnostic data with content, web_sources, youtube_videos, etc.
        - null if no data is available
    """
    try:
        logger.info("📊 API: Attempting to retrieve diagnostic data...")
        diagnostic_data = get_latest_diagnostic_data()
        
        if diagnostic_data is None:
            logger.info("📭 API: No diagnostic data available from get_latest_diagnostic_data()")
            return JSONResponse(
                status_code=204,  # No Content
                content={"message": "No diagnostic data available", "data": None}
            )
        
        logger.info(f"📊 API: Returning diagnostic data - {len(diagnostic_data.get('web_sources', []))} web sources, {len(diagnostic_data.get('youtube_videos', []))} videos")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Diagnostic data retrieved successfully",
                "data": diagnostic_data
            }
        )
        
    except Exception as e:
        logger.error(f"❌ API Error retrieving diagnostic data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.delete("/api/diagnostic-data")
async def clear_diagnostic_data_endpoint():
    """
    Clear the stored diagnostic data.
    Useful for cleanup or testing purposes.
    """
    try:
        clear_diagnostic_data()
        logger.info("🧹 API: Diagnostic data cleared")
        
        return JSONResponse(
            status_code=200,
            content={"message": "Diagnostic data cleared successfully"}
        )
        
    except Exception as e:
        logger.error(f"❌ API Error clearing diagnostic data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """Detailed health check with diagnostic system status"""
    try:
        # Test if we can access diagnostic functions
        diagnostic_data = get_latest_diagnostic_data()
        has_data = diagnostic_data is not None
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "diagnostic_system": "operational",
                "has_diagnostic_data": has_data,
                "timestamp": "2025-10-29T11:39:49.073306"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return JSONResponse(
            status_code=503,  # Service Unavailable
            content={
                "status": "unhealthy",
                "diagnostic_system": "error",
                "error": str(e),
                "timestamp": "2025-10-29T11:39:49.073306"
            }
        )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Allion Diagnostic API server...")
    logger.info("📊 Endpoints available:")
    logger.info("  GET  /                      - Root health check")
    logger.info("  GET  /api/diagnostic-data   - Get latest diagnostic data") 
    logger.info("  DELETE /api/diagnostic-data - Clear diagnostic data")
    logger.info("  GET  /api/health           - Detailed health check")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",  # Allow external connections
        port=8001,
        reload=True,
        log_level="info"
    )
