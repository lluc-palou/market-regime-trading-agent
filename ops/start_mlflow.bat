@echo off
setlocal

set "MLFLOW_EXE=mlflow"
set "MLFLOW_DIR=C:\mlflow"
set "MLFLOW_BACKEND_STORE_URI=sqlite:///C:/mlflow/mlflow.db"
set "MLFLOW_ARTIFACT_ROOT=file:///C:/mlflow/mlruns"

rem Ensure folders exist
if not exist "%MLFLOW_DIR%" mkdir "%MLFLOW_DIR%"
if not exist "%MLFLOW_DIR%\mlruns" mkdir "%MLFLOW_DIR%\mlruns"

echo Starting MLflow tracking server...
"%MLFLOW_EXE%" server ^
  --backend-store-uri "%MLFLOW_BACKEND_STORE_URI%" ^
  --default-artifact-root "%MLFLOW_ARTIFACT_ROOT%" ^
  --host 127.0.0.1 ^
  --port 5000
echo.

endlocal