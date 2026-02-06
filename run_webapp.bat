@echo off
echo ==========================================
echo  LLMSQL2 Web App - Starting in Docker
echo ==========================================
echo.

REM Check if container is running
docker ps | findstr "llmsql2-app" >nul
if errorlevel 1 (
    echo Container not running. Starting Docker services...
    docker-compose up -d
    echo Waiting for container to be ready...
    timeout /t 5 /nobreak >nul
)

echo.
echo Starting web app...
echo Open http://localhost:5000 in your browser
echo.
echo Press Ctrl+C to stop the web app
echo.

docker exec -it llmsql2-app python /app/src/web_app_docker.py
