#!/bin/bash
# =================================================================
# Test Runner Script with Multiple Modes
# Purpose: Run tests with different configurations
# Author: Carlos Daniel Jiménez
# =================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default mode
MODE="${1:-quick}"

echo -e "${BLUE}=========================================="
echo "  MLOps Pipeline Test Runner"
echo -e "==========================================${NC}\n"

case "$MODE" in
    quick)
        echo -e "${YELLOW}Running QUICK tests (fast unit tests only)${NC}"
        echo "Excludes: slow tests, integration tests"
        echo "Expected time: ~10-30 seconds"
        echo ""
        pytest -v -m "not slow and not integration" tests/
        ;;

    full)
        echo -e "${YELLOW}Running FULL test suite (including slow tests)${NC}"
        echo "Excludes: integration tests (require network)"
        echo "Expected time: ~1-5 minutes"
        echo ""
        pytest -v -m "not integration" tests/
        ;;

    integration)
        echo -e "${YELLOW}Running tests WITH integration tests${NC}"
        echo "Includes: ALL tests + real network requests"
        echo "Expected time: ~2-10 minutes"
        echo "Requirements: Internet connection"
        echo ""
        pytest -v --run-integration tests/
        ;;

    performance)
        echo -e "${YELLOW}Running PERFORMANCE benchmarks${NC}"
        echo "Shows timing information for each test"
        echo "Expected time: ~2-5 minutes"
        echo ""
        pytest -v -m performance -s tests/
        ;;

    coverage)
        echo -e "${YELLOW}Running tests WITH coverage report${NC}"
        echo "Generates HTML coverage report"
        echo "Expected time: ~1-5 minutes"
        echo ""
        pytest --cov=src --cov-report=html --cov-report=term tests/
        echo ""
        echo -e "${GREEN}Coverage report generated: htmlcov/index.html${NC}"
        echo "Open with: open htmlcov/index.html"
        ;;

    realistic)
        echo -e "${YELLOW}Running REALISTIC tests only${NC}"
        echo "Tests that use real data and real edge cases"
        echo "Expected time: ~30 seconds"
        echo ""
        pytest -v tests/test_*_realistic.py
        ;;

    watch)
        echo -e "${YELLOW}Running tests in WATCH mode${NC}"
        echo "Tests will re-run on file changes"
        echo "Press Ctrl+C to stop"
        echo ""
        pytest-watch -v -m "not slow and not integration" tests/
        ;;

    help|--help|-h)
        echo "Usage: ./run_tests.sh [MODE]"
        echo ""
        echo "Available modes:"
        echo "  quick        - Fast unit tests only (default)"
        echo "  full         - All tests except integration"
        echo "  integration  - All tests including real downloads"
        echo "  performance  - Performance benchmarks only"
        echo "  coverage     - Generate coverage report"
        echo "  realistic    - Only realistic test files"
        echo "  watch        - Run tests on file changes"
        echo "  help         - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh                 # Quick tests"
        echo "  ./run_tests.sh full            # All tests"
        echo "  ./run_tests.sh integration     # With real downloads"
        echo "  ./run_tests.sh coverage        # With coverage"
        echo ""
        exit 0
        ;;

    *)
        echo -e "${RED}Error: Unknown mode '$MODE'${NC}"
        echo "Run './run_tests.sh help' for usage"
        exit 1
        ;;
esac

echo ""
if [ $? -eq 0 ]; then
    echo -e "${GREEN}=========================================="
    echo "  ✅ ALL TESTS PASSED"
    echo -e "==========================================${NC}"
else
    echo -e "${RED}=========================================="
    echo "  ❌ TESTS FAILED"
    echo -e "==========================================${NC}"
    exit 1
fi
