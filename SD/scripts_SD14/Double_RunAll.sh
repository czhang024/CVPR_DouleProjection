#!/bin/bash

# Master script to run all Double* scripts for SD 1.4 sequentially
# Each script dumps results to its own log file

set -e  # Exit on error

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "Running all Double* scripts for SD 1.4"
echo "Started at: $(date)"
echo "=========================================="

# Run each Double* script sequentially with nohup and separate log files
echo "Starting Double_CassettePlayer..."
nohup sh scripts_SD14/Double_CassettePlayer.sh > logs/Double_CassettePlayer_SD14.log 2>&1
echo "✓ Double_CassettePlayer completed"

echo "Starting Double_ChainSaw..."
nohup sh scripts_SD14/Double_ChainSaw.sh > logs/Double_ChainSaw_SD14.log 2>&1
echo "✓ Double_ChainSaw completed"

echo "Starting Double_Church..."
nohup sh scripts_SD14/Double_Church.sh > logs/Double_Church_SD14.log 2>&1
echo "✓ Double_Church completed"

echo "Starting Double_EnglishSpringer..."
nohup sh scripts_SD14/Double_EnglishSpringer.sh > logs/Double_EnglishSpringer_SD14.log 2>&1
echo "✓ Double_EnglishSpringer completed"

echo "Starting Double_FrenchHorn..."
nohup sh scripts_SD14/Double_FrenchHorn.sh > logs/Double_FrenchHorn_SD14.log 2>&1
echo "✓ Double_FrenchHorn completed"

echo "Starting Double_GarbageTruck..."
nohup sh scripts_SD14/Double_GarbageTruck.sh > logs/Double_GarbageTruck_SD14.log 2>&1
echo "✓ Double_GarbageTruck completed"

echo "Starting Double_GasPump..."
nohup sh scripts_SD14/Double_GasPump.sh > logs/Double_GasPump_SD14.log 2>&1
echo "✓ Double_GasPump completed"

echo "Starting Double_GolfBall..."
nohup sh scripts_SD14/Double_GolfBall.sh > logs/Double_GolfBall_SD14.log 2>&1
echo "✓ Double_GolfBall completed"

echo "Starting Double_Parachute..."
nohup sh scripts_SD14/Double_Parachute.sh > logs/Double_Parachute_SD14.log 2>&1
echo "✓ Double_Parachute completed"

echo "Starting Double_Tench..."
nohup sh scripts_SD14/Double_Tench.sh > logs/Double_Tench_SD14.log 2>&1
echo "✓ Double_Tench completed"

echo "=========================================="
echo "All Double* scripts completed!"
echo "Finished at: $(date)"
echo "=========================================="
echo ""
echo "Log files are saved in logs/ directory:"
echo "  - logs/Double_CassettePlayer_SD14.log"
echo "  - logs/Double_ChainSaw_SD14.log"
echo "  - logs/Double_Church_SD14.log"
echo "  - logs/Double_EnglishSpringer_SD14.log"
echo "  - logs/Double_FrenchHorn_SD14.log"
echo "  - logs/Double_GarbageTruck_SD14.log"
echo "  - logs/Double_GasPump_SD14.log"
echo "  - logs/Double_GolfBall_SD14.log"
echo "  - logs/Double_Parachute_SD14.log"
echo "  - logs/Double_Tench_SD14.log"
echo ""
echo "Model weights saved in models/ directory with unique names:"
echo "  - models/UCE_double_proxy_CassettePlayer/"
echo "  - models/UCE_double_proxy_ChainSaw/"
echo "  - models/UCE_double_proxy_Church/"
echo "  - models/UCE_double_proxy_EnglishSpringer/"
echo "  - models/UCE_double_proxy_FrenchHorn/"
echo "  - models/UCE_double_proxy_GarbageTruck/"
echo "  - models/UCE_double_proxy_GasPump/"
echo "  - models/UCE_double_proxy_GolfBall/"
echo "  - models/UCE_double_proxy_Parachute/"
echo "  - models/UCE_double_proxy_Tench/"
echo ""
echo "Generated images saved in concept-specific directories:"
echo "  - generated_Double_CassettePlayer/"
echo "  - generated_Double_ChainSaw/"
echo "  - generated_Double_Church/"
echo "  - generated_Double_EnglishSpringer/"
echo "  - generated_Double_FrenchHorn/"
echo "  - generated_Double_GarbageTruck/"
echo "  - generated_Double_GasPump/"
echo "  - generated_Double_GolfBall/"
echo "  - generated_Double_Parachute/"
echo "  - generated_Double_Tench/"
echo ""
echo "Comparison results saved as concept-specific log files:"
echo "  - DoubleResult_CassettePlayer.log"
echo "  - DoubleResult_ChainSaw.log"
echo "  - DoubleResult_Church.log"
echo "  - DoubleResult_EnglishSpringer.log"
echo "  - DoubleResult_FrenchHorn.log"
echo "  - DoubleResult_GarbageTruck.log"
echo "  - DoubleResult_GasPump.log"
echo "  - DoubleResult_GolfBall.log"
echo "  - DoubleResult_Parachute.log"
echo "  - DoubleResult_Tench.log"

