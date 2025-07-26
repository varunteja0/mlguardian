from fastapi import FastAPI

app = FastAPI(title="MLGuardian Core Engine")

@app.get("/")
async def root():
    return {"message": "MLGuardian Core Engine Running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "mlguardian-core"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)