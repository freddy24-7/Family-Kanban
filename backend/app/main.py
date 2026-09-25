from fastapi import FastAPI

app = FastAPI(title="Family Kanban API")


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> dict[str, str]:
    # Static on purpose: uptime monitors must not trigger DB or model work.
    return {"status": "ok"}
