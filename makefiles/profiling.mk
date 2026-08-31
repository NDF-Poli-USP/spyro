# ============================================================
# PROFILING TARGETS
# ============================================================

.PHONY: profile-cpu profile-memory profile-cpu-mpi profile-memory-mpi profile-all profile-all-mpi profile-clean profile-help

# Directory for all profiling output
PROFILERS_DIR := profilers

# Validation function to check if FILE is provided
define check_file
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: FILE is required. Usage: make $(1) FILE=path/to/file.py [FUNCTION=def_name]"; \
		echo "Example: make $(1) FILE=path/to/file.py"; \
		echo "Example: make $(1) FILE=path/to/file.py FUNCTION=def_name"; \
		exit 1; \
	fi
	@if [ ! -f "$(FILE)" ]; then \
		echo "ERROR: File '$(FILE)' not found!"; \
		exit 1; \
	fi
endef

# Shared values used by the profiling targets
TIMESTAMP := $(shell date +"%Y%m%d_%H%M%S")
MPI_NPROCS ?= 4
MPI_CMD ?= mpiexec -n $(MPI_NPROCS)
PROFILE_ROOT := $(subst /,_,$(FILE:.py=))
PROFILE_SUFFIX := $(if $(FUNCTION),_$(FUNCTION),)
PROFILE_SESSION := $(PROFILERS_DIR)/profile_$(PROFILE_ROOT)$(PROFILE_SUFFIX)_$(TIMESTAMP).pyisession
PROFILE_HTML := $(PROFILERS_DIR)/cpu_profile_$(PROFILE_ROOT)$(PROFILE_SUFFIX)_$(TIMESTAMP).html
PROFILE_DAT := $(PROFILERS_DIR)/mprofile_$(PROFILE_ROOT)$(PROFILE_SUFFIX)_$(TIMESTAMP).dat
PROFILE_PNG := $(PROFILERS_DIR)/memory_profile_$(PROFILE_ROOT)$(PROFILE_SUFFIX)_$(TIMESTAMP).png

# ============================================================
# PYINSTRUMENT (CPU Profiling)
# ============================================================

# CPU profiling with pyinstrument - always generates HTML
# Usage: make profile-cpu FILE=path/to/file.py [FUNCTION=def_name]
profile-cpu:
	@mkdir -p $(PROFILERS_DIR)
	$(call check_file,profile-cpu)
	@echo "========================================="
	@echo "CPU Profiling with pyinstrument"
	@echo "========================================="
	@echo "File: $(FILE)"
	@if [ -n "$(FUNCTION)" ]; then echo "Function: $(FUNCTION)"; fi
	@echo ""
	@python3 -m pip show pyinstrument > /dev/null 2>&1 || { echo "pyinstrument is not available. Run 'make install-dev' first."; exit 1; }
	@if [ -n "$(FUNCTION)" ]; then \
		echo "Running: pyinstrument -i 0.05 -r pyisession -o $(PROFILE_SESSION) -m pytest $(FILE)::$(FUNCTION) -s"; \
		pyinstrument -i 0.05 -r pyisession -o $(PROFILE_SESSION) -m pytest $(FILE)::$(FUNCTION) -s; \
	else \
		echo "Running: pyinstrument -i 0.05 -r pyisession -o $(PROFILE_SESSION) $(FILE)"; \
		pyinstrument -i 0.05 -r pyisession -o $(PROFILE_SESSION) $(FILE); \
	fi
	@echo ""
	@echo "Generating HTML report..."
	@pyinstrument --load $(PROFILE_SESSION) -r html -o $(PROFILE_HTML)
	@echo ""
	@echo "✓ CPU profile saved to: $(PROFILE_HTML)"
	@echo "✓ Session saved to: $(PROFILE_SESSION)"
	@echo "To view: firefox $(PROFILE_HTML)"

# ============================================================
# MPROF (Memory Profiling)
# ============================================================

# Memory profiling with mprof - always generates PNG
# Usage: make profile-memory FILE=path/to/file.py [FUNCTION=def_name]
profile-memory:
	@mkdir -p $(PROFILERS_DIR)
	$(call check_file,profile-memory)
	@echo "========================================="
	@echo "Memory Profiling with mprof"
	@echo "========================================="
	@echo "File: $(FILE)"
	@if [ -n "$(FUNCTION)" ]; then echo "Function: $(FUNCTION)"; fi
	@echo ""
	@python3 -m pip show memory_profiler > /dev/null 2>&1 || { echo "memory_profiler is not available. Run 'make install-dev' first."; exit 1; }
	@python3 -m pip show psutil > /dev/null 2>&1 || { echo "psutil is not available. Run 'make install-dev' first."; exit 1; }
	@rm -f $(PROFILERS_DIR)/mprofile_*.dat 2>/dev/null || true
	@echo "Running memory profile..."
	@if [ -n "$(FUNCTION)" ]; then \
		echo "Running: mprof run -o $(PROFILE_DAT) pytest $(FILE)::$(FUNCTION) -s"; \
		mprof run -o $(PROFILE_DAT) pytest $(FILE)::$(FUNCTION) -s; \
	else \
		echo "Running: mprof run -o $(PROFILE_DAT) python3 $(FILE)"; \
		mprof run -o $(PROFILE_DAT) python3 $(FILE); \
	fi
	@echo ""
	@echo "Generating memory plot..."
	@mprof plot -o $(PROFILE_PNG) $(PROFILE_DAT)
	@echo ""
	@echo "✓ Memory profile saved to: $(PROFILE_PNG)"
	@echo "✓ Data saved to: $(PROFILE_DAT)"
	@echo "To view: eog $(PROFILE_PNG)"

# ============================================================
# MPI PROFILING
# ============================================================

# CPU profiling for an MPI run
# Usage: make profile-cpu-mpi FILE=path/to/file.py [FUNCTION=def_name] [MPI_NPROCS=4]
profile-cpu-mpi:
	@mkdir -p $(PROFILERS_DIR)
	$(call check_file,profile-cpu-mpi)
	@echo "========================================="
	@echo "CPU Profiling with MPI"
	@echo "========================================="
	@echo "File: $(FILE)"
	@if [ -n "$(FUNCTION)" ]; then echo "Function: $(FUNCTION)"; fi
	@echo "MPI command: $(MPI_CMD)"
	@echo ""
	@python3 -m pip show pyinstrument > /dev/null 2>&1 || { echo "pyinstrument is not available. Run 'make install-dev' first."; exit 1; }
	@if [ -n "$(FUNCTION)" ]; then \
		echo "Running: $(MPI_CMD) python3 -m pyinstrument -i 0.05 -r pyisession -o $(PROFILE_SESSION) -m pytest $(FILE)::$(FUNCTION) -s"; \
		$(MPI_CMD) python3 -m pyinstrument -i 0.05 -r pyisession -o $(PROFILE_SESSION) -m pytest $(FILE)::$(FUNCTION) -s; \
	else \
		echo "Running: $(MPI_CMD) python3 -m pyinstrument -i 0.05 -r pyisession -o $(PROFILE_SESSION) $(FILE)"; \
		$(MPI_CMD) python3 -m pyinstrument -i 0.05 -r pyisession -o $(PROFILE_SESSION) $(FILE); \
	fi
	@echo ""
	@echo "Generating HTML report..."
	@pyinstrument --load $(PROFILE_SESSION) -r html -o $(PROFILE_HTML)
	@echo ""
	@echo "✓ CPU profile saved to: $(PROFILE_HTML)"
	@echo "✓ Session saved to: $(PROFILE_SESSION)"

# Memory profiling for an MPI run
# Usage: make profile-memory-mpi FILE=path/to/file.py [FUNCTION=def_name] [MPI_NPROCS=4]
profile-memory-mpi:
	@mkdir -p $(PROFILERS_DIR)
	$(call check_file,profile-memory-mpi)
	@echo "========================================="
	@echo "Memory Profiling with MPI"
	@echo "========================================="
	@echo "File: $(FILE)"
	@if [ -n "$(FUNCTION)" ]; then echo "Function: $(FUNCTION)"; fi
	@echo "MPI command: $(MPI_CMD)"
	@echo ""
	@python3 -m pip show memory_profiler > /dev/null 2>&1 || { echo "memory_profiler is not available. Run 'make install-dev' first."; exit 1; }
	@python3 -m pip show psutil > /dev/null 2>&1 || { echo "psutil is not available. Run 'make install-dev' first."; exit 1; }
	@rm -f $(PROFILERS_DIR)/mprofile_*.dat 2>/dev/null || true
	@echo "Running memory profile..."
	@if [ -n "$(FUNCTION)" ]; then \
		echo "Running: mprof run -o $(PROFILE_DAT) -- sh -c '$(MPI_CMD) python3 -m memory_profiler -m pytest $(FILE)::$(FUNCTION) -s'"; \
		mprof run -o $(PROFILE_DAT) -- sh -c '$(MPI_CMD) python3 -m memory_profiler -m pytest $(FILE)::$(FUNCTION) -s'; \
	else \
		echo "Running: mprof run -o $(PROFILE_DAT) -- sh -c '$(MPI_CMD) python3 -m memory_profiler $(FILE)'"; \
		mprof run -o $(PROFILE_DAT) -- sh -c '$(MPI_CMD) python3 -m memory_profiler $(FILE)'; \
	fi
	@echo ""
	@echo "Generating memory plot..."
	@mprof plot -o $(PROFILE_PNG) $(PROFILE_DAT)
	@echo ""
	@echo "✓ Memory profile saved to: $(PROFILE_PNG)"
	@echo "✓ Data saved to: $(PROFILE_DAT)"

# ============================================================
# COMBINED PROFILING
# ============================================================

# Run both CPU and memory profiling
# Usage: make profile-all FILE=path/to/file.py [FUNCTION=def_name]
profile-all:
	$(call check_file,profile-all)
	@echo "========================================="
	@echo "Running complete profiling suite"
	@echo "========================================="
	@echo "File: $(FILE)"
	@if [ -n "$(FUNCTION)" ]; then echo "Function: $(FUNCTION)"; fi
	@echo ""
	@$(MAKE) profile-cpu FILE="$(FILE)" FUNCTION="$(FUNCTION)"
	@echo ""
	@$(MAKE) profile-memory FILE="$(FILE)" FUNCTION="$(FUNCTION)"
	@echo ""
	@echo "========================================="
	@echo "✓ All profiles complete"
	@echo "========================================="

# Run both CPU and memory profiling for an MPI job
# Usage: make profile-all-mpi FILE=path/to/file.py [FUNCTION=def_name] [MPI_NPROCS=4]
profile-all-mpi:
	$(call check_file,profile-all-mpi)
	@echo "========================================="
	@echo "Running complete MPI profiling suite"
	@echo "========================================="
	@echo "File: $(FILE)"
	@if [ -n "$(FUNCTION)" ]; then echo "Function: $(FUNCTION)"; fi
	@echo "MPI command: $(MPI_CMD)"
	@echo ""
	@$(MAKE) profile-cpu-mpi FILE="$(FILE)" FUNCTION="$(FUNCTION)" MPI_NPROCS="$(MPI_NPROCS)"
	@echo ""
	@$(MAKE) profile-memory-mpi FILE="$(FILE)" FUNCTION="$(FUNCTION)" MPI_NPROCS="$(MPI_NPROCS)"
	@echo ""
	@echo "========================================="
	@echo "✓ All MPI profiles complete"
	@echo "========================================="

# ============================================================
# UTILITY TARGETS
# ============================================================

# Clean profiling artifacts
profile-clean:
	@echo "Cleaning profiling artifacts..."
	@rm -rf $(PROFILERS_DIR)/* 2>/dev/null || true
	@rm -rf mprofile_*.dat 2>/dev/null || true
	@mprof clean 2>/dev/null || true
	@echo "✓ Cleaned all profiling files"

# Help for profiling targets
profile-help:
	@echo "========================================="
	@echo "Profiling Targets (FILE is REQUIRED)"
	@echo "========================================="
	@echo ""
	@echo "All profiling results are saved in the 'profilers/' directory"
	@echo ""
	@echo "CPU Profiling (pyinstrument):"
	@echo "  make profile-cpu FILE=path/to/file.py [FUNCTION=def_name]"
	@echo "    - Generates: profilers/cpu_profile_*.html"
	@echo "    - Saves session: profilers/profile_*.pyisession"
	@echo ""
	@echo "Memory Profiling (mprof):"
	@echo "  make profile-memory FILE=path/to/file.py [FUNCTION=def_name]"
	@echo "    - Generates: profilers/memory_profile_*.png"
	@echo ""
	@echo "Combined Profiling:"
	@echo "  make profile-all FILE=path/to/file.py [FUNCTION=def_name]"
	@echo "    - Runs both CPU and memory profiling"
	@echo ""
	@echo "MPI Profiling:"
	@echo "  make profile-cpu-mpi FILE=path/to/file.py [FUNCTION=def_name] [MPI_NPROCS=4]"
	@echo "    - Runs CPU profiling through mpiexec"
	@echo "  make profile-memory-mpi FILE=path/to/file.py [FUNCTION=def_name] [MPI_NPROCS=4]"
	@echo "    - Runs memory profiling through mpiexec"
	@echo "  make profile-all-mpi FILE=path/to/file.py [FUNCTION=def_name] [MPI_NPROCS=4]"
	@echo "    - Runs both MPI profiling modes"
	@echo ""
	@echo "Utilities:"
	@echo "  make profile-clean    - Remove all profiling artifacts"
	@echo "  make profile-help     - Show this help"
	@echo ""
	@echo "========================================="
	@echo "Examples:"
	@echo "========================================="
	@echo "  # Profile a full test file"
	@echo "  make profile-cpu FILE=tests/on_one_core/test_eikonal.py"
	@echo ""
	@echo "  # Profile a specific test function"
	@echo "  make profile-cpu FILE=tests/on_one_core/test_eikonal.py FUNCTION=test_loop_eikonal_2d"
	@echo ""
	@echo "  # Memory profile a specific test"
	@echo "  make profile-memory FILE=tests/on_one_core/test_eikonal.py FUNCTION=test_loop_eikonal_2d"
	@echo ""
	@echo "  # Run both full CPU and memory profiles"
	@echo "  make profile-all FILE=tests/on_one_core/test_eikonal.py"
	@echo ""
	@echo "  # Run both specific CPU and memory profiles"
	@echo "  make profile-all FILE=tests/on_one_core/test_eikonal.py FUNCTION=test_loop_eikonal_2d"
	@echo ""
	@echo "  # Profile an MPI run"
	@echo "  make profile-all-mpi FILE=tests/on_one_core/test_eikonal.py MPI_NPROCS=4"
	@echo ""
	@echo "  # Clean all profiling files"
	@echo "  make profile-clean"
