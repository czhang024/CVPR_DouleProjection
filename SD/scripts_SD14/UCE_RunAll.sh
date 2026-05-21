#!/bin/bash

# Master script to run all UCE* scripts for SD 1.4 sequentially
# Each script dumps results to its own log file

set -e  # Exit on error

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "Running all UCE* scripts for SD 1.4"
echo "Started at: $(date)"
echo "=========================================="

# Run each UCE* script sequentially with nohup and separate log files
echo "Starting UCE_CassettePlayer..."
nohup sh scripts_SD14/UCE_CassettePlayer.sh > logs/UCE_CassettePlayer_SD14.log 2>&1
echo "✓ UCE_CassettePlayer completed"

echo "Starting UCE_ChainSaw..."
nohup sh scripts_SD14/UCE_ChainSaw.sh > logs/UCE_ChainSaw_SD14.log 2>&1
echo "✓ UCE_ChainSaw completed"

echo "Starting UCE_Church..."
nohup sh scripts_SD14/UCE_Church.sh > logs/UCE_Church_SD14.log 2>&1
echo "✓ UCE_Church completed"

echo "Starting UCE_EnglishSpringer..."
nohup sh scripts_SD14/UCE_EnglishSpringer.sh > logs/UCE_EnglishSpringer_SD14.log 2>&1
echo "✓ UCE_EnglishSpringer completed"

echo "Starting UCE_FrenchHorn..."
nohup sh scripts_SD14/UCE_FrenchHorn.sh > logs/UCE_FrenchHorn_SD14.log 2>&1
echo "✓ UCE_FrenchHorn completed"

echo "Starting UCE_GarbageTruck..."
nohup sh scripts_SD14/UCE_GarbageTruck.sh > logs/UCE_GarbageTruck_SD14.log 2>&1
echo "✓ UCE_GarbageTruck completed"

echo "Starting UCE_GasPump..."
nohup sh scripts_SD14/UCE_GasPump.sh > logs/UCE_GasPump_SD14.log 2>&1
echo "✓ UCE_GasPump completed"

echo "Starting UCE_GolfBall..."
nohup sh scripts_SD14/UCE_GolfBall.sh > logs/UCE_GolfBall_SD14.log 2>&1
echo "✓ UCE_GolfBall completed"

echo "Starting UCE_Parachute..."
nohup sh scripts_SD14/UCE_Parachute.sh > logs/UCE_Parachute_SD14.log 2>&1
echo "✓ UCE_Parachute completed"

echo "Starting UCE_Tench..."
nohup sh scripts_SD14/UCE_Tench.sh > logs/UCE_Tench_SD14.log 2>&1
echo "✓ UCE_Tench completed"

echo "=========================================="
echo "All UCE* scripts completed!"
echo "Finished at: $(date)"
echo "=========================================="
echo ""
echo "Log files are saved in logs/ directory:"
echo "  - logs/UCE_CassettePlayer_SD14.log"
echo "  - logs/UCE_ChainSaw_SD14.log"
echo "  - logs/UCE_Church_SD14.log"
echo "  - logs/UCE_EnglishSpringer_SD14.log"
echo "  - logs/UCE_FrenchHorn_SD14.log"
echo "  - logs/UCE_GarbageTruck_SD14.log"
echo "  - logs/UCE_GasPump_SD14.log"
echo "  - logs/UCE_GolfBall_SD14.log"
echo "  - logs/UCE_Parachute_SD14.log"
echo "  - logs/UCE_Tench_SD14.log"
echo ""
echo "Model weights saved in models/ directory with unique names:"
echo "  - models/UCE_original_CassettePlayer/"
echo "  - models/UCE_original_ChainSaw/"
echo "  - models/UCE_original_Church/"
echo "  - models/UCE_original_EnglishSpringer/"
echo "  - models/UCE_original_FrenchHorn/"
echo "  - models/UCE_original_GarbageTruck/"
echo "  - models/UCE_original_GasPump/"
echo "  - models/UCE_original_GolfBall/"
echo "  - models/UCE_original_Parachute/"
echo "  - models/UCE_original_Tench/"
echo ""
echo "Generated images saved in concept-specific directories:"
echo "  - generated_UCE_CassettePlayer/"
echo "  - generated_UCE_ChainSaw/"
echo "  - generated_UCE_Church/"
echo "  - generated_UCE_EnglishSpringer/"
echo "  - generated_UCE_FrenchHorn/"
echo "  - generated_UCE_GarbageTruck/"
echo "  - generated_UCE_GasPump/"
echo "  - generated_UCE_GolfBall/"
echo "  - generated_UCE_Parachute/"
echo "  - generated_UCE_Tench/"
echo ""
echo "Comparison results saved as concept-specific log files:"
echo "  - UCEResult_CassettePlayer.log"
echo "  - UCEResult_ChainSaw.log"
echo "  - UCEResult_Church.log"
echo "  - UCEResult_EnglishSpringer.log"
echo "  - UCEResult_FrenchHorn.log"
echo "  - UCEResult_GarbageTruck.log"
echo "  - UCEResult_GasPump.log"
echo "  - UCEResult_GolfBall.log"
echo "  - UCEResult_Parachute.log"
echo "  - UCEResult_Tench.log"

