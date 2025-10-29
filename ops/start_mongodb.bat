@echo off
setlocal

set "MONGO_EXE=C:\Program Files\MongoDB\Server\8.0\bin\mongod.exe"
set "MONGO_DBPATH=C:\data\db"

rem Ensure folders exists
if not exist "%MONGO_DBPATH%" mkdir "%MONGO_DBPATH%"

echo Starting MongoDB server...
"%MONGO_EXE%" --dbpath "%MONGO_DBPATH%"
echo.

endlocal