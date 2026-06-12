# TODO — Fix API communication + ensure end-to-end dashboard updates

## Plan (approved)
- Priority: make backend not-running situation handled permanently (user-visible behavior), then fix duplicate chart rendering / polling.

## Steps
1. Inspect `backend/requirements.txt` and `README.md` to determine correct startup command.
2. Run backend smoke test (start Flask server in test mode or use pytest) to confirm baseline.
3. Fix frontend `SpamAPI.resolveApiBase()` and `fetchApi()` so the UI clearly and permanently handles “backend not running”:
   - Provide single, stable error state
   - Prevent repeated polling attempts when backend is unreachable
4. Fix duplicate chart rendering by destroying/disposing charts before re-rendering in live polling.
5. Add request timeouts + abort logic to avoid hanging fetches.
6. Run `pytest` to verify all endpoints.
7. Produce final report: root cause per issue, files modified, code changes, request/response payload snapshots from tests, before vs after table.

