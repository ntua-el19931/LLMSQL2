@echo off
REM LLMSQL2 Docker Helper Script for Windows

echo ============================================
echo LLMSQL2 - Text-to-SQL Docker Application
echo ============================================

if "%1"=="" goto help
if "%1"=="build" goto build
if "%1"=="up" goto up
if "%1"=="down" goto down
if "%1"=="logs" goto logs
if "%1"=="shell" goto shell
if "%1"=="clean" goto clean
if "%1"=="gpu" goto gpu
goto help

:build
echo Building LLMSQL2 containers...
docker-compose build
goto end

:up
echo Starting LLMSQL2...
docker-compose up -d
echo.
echo LLMSQL2 is running!
echo Jupyter Lab: http://localhost:8888
echo PostgreSQL:  localhost:5432
goto end

:down
echo Stopping LLMSQL2...
docker-compose down
goto end

:logs
echo Showing logs...
docker-compose logs -f
goto end

:shell
echo Opening shell in app container...
docker-compose exec app /bin/bash
goto end

:clean
echo Stopping and removing all containers and volumes...
docker-compose down -v
docker rmi llmsql2-app 2>nul
echo Cleaned up!
goto end

:gpu
echo Building with GPU support...
docker-compose -f docker-compose.gpu.yml up -d
goto end

:help
echo.
echo Usage: run.bat [command]
echo.
echo Commands:
echo   build  - Build Docker images
echo   up     - Start all services
echo   down   - Stop all services
echo   logs   - Show container logs
echo   shell  - Open shell in app container
echo   clean  - Remove containers and volumes
echo   gpu    - Start with GPU support (NVIDIA)
echo.

:end
