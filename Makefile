VERSION=$(shell python3 -c "from configparser import ConfigParser; p = ConfigParser(); p.read('setup.cfg'); print(p['metadata']['version'])")

default:
	@echo "\"make publish\"?"

tag:

	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "main" ]; then exit 1; fi
	curl -H "Authorization: token `cat $(HOME)/.github-access-token`" -d '{"tag_name": "v$(VERSION)"}' https://api.github.com/repos/krober10nd/spyro/releases

upload:
	# Make sure we're on the main branch
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "main" ]; then exit 1; fi

	rm -rf dist/*
	python3 setup.py sdist bdist_wheel
	twine upload dist/*

publish: tag upload

clean-pyc:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf
	@rm -rf build/*
	@rm -rf spyro.egg-info/
	@rm -rf dist/

clean-root:
	@rm -f *.msh *.vtk *.png *.vtu *.pvtu *.pvd *.npy *.pdf *.dat *.segy *.hdf5
	@rm -rf asn*/ bsn*/

clean-data:
	@rm -f shots/*.dat
	@rm -f *.txt

clean-output:
	@rm -rf velocity_models/test*
	@rm -rf results/*
	@rm -rf control_*/ gradient*/ initial_velocity_model/ output*/ vp_end*/ test_debug*/

clean: clean-pyc clean-root clean-data clean-output

format:
	autopep8 --in-place --global-config setup.cfg --recursive .

lint:
	flake8 setup.py spyro/ tests/integration/*.py tests/on_one_core/*.py

# ============================================================
# PROFILING TARGETS
# ============================================================

# Directory for all profiling output
PROFILERS_DIR := profilers
$(shell mkdir -p $(PROFILERS_DIR))

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

# Timestamp for unique filenames
TIMESTAMP := $(shell date +"%Y%m%d_%H%M%S")

# ============================================================
# PYINSTRUMENT (CPU Profiling)
# ============================================================

# CPU profiling with pyinstrument - always generates HTML
# Usage: make profile-cpu FILE=path/to/file.py [FUNCTION=def_name]
profile-cpu:
	$(call check_file,profile-cpu)
	@echo "========================================="
	@echo "CPU Profiling with pyinstrument"
	@echo "========================================="
	@echo "File: $(FILE)"
	@if [ -n "$(FUNCTION)" ]; then echo "Function: $(FUNCTION)"; fi
	@echo ""
	# Check if pyinstrument is installed
	@pip show pyinstrument > /dev/null 2>&1 || (echo "Installing pyinstrument..." && pip install pyinstrument)
	# Generate session name based on file and function
	$(eval SESSION_NAME := $(PROFILERS_DIR)/profile_$(subst /,_,$(FILE:.py=))$(if $(FUNCTION),_$(FUNCTION),)_$(TIMESTAMP).pyisession)
	$(eval HTML_NAME := $(PROFILERS_DIR)/cpu_profile_$(subst /,_,$(FILE:.py=))$(if $(FUNCTION),_$(FUNCTION),)_$(TIMESTAMP).html)
	# Run profiling
	@if [ -n "$(FUNCTION)" ]; then \
		echo "Running: pyinstrument -i 0.05 -r pyisession -o $(SESSION_NAME) -m pytest $(FILE)::$(FUNCTION) -s"; \
		pyinstrument -i 0.05 -r pyisession -o $(SESSION_NAME) -m pytest $(FILE)::$(FUNCTION) -s; \
	else \
		echo "Running: pyinstrument -i 0.05 -r pyisession -o $(SESSION_NAME) $(FILE)"; \
		pyinstrument -i 0.05 -r pyisession -o $(SESSION_NAME) $(FILE); \
	fi
	# Generate HTML from session
	@echo ""
	@echo "Generating HTML report..."
	@pyinstrument --load $(SESSION_NAME) -r html -o $(HTML_NAME)
	@echo ""
	@echo "✓ CPU profile saved to: $(HTML_NAME)"
	@echo "✓ Session saved to: $(SESSION_NAME)"
	@echo ""
	@echo "To view: firefox $(HTML_NAME)"
	@echo "To generate other formats: pyinstrument --load $(SESSION_NAME) -r text -o $(PROFILERS_DIR)/profile.txt"

# ============================================================
# MPROF (Memory Profiling)
# ============================================================

# Memory profiling with mprof - always generates PNG
# Usage: make profile-memory FILE=path/to/file.py [FUNCTION=def_name]
profile-memory:
	$(call check_file,profile-memory)
	@echo "========================================="
	@echo "Memory Profiling with mprof"
	@echo "========================================="
	@echo "File: $(FILE)"
	@if [ -n "$(FUNCTION)" ]; then echo "Function: $(FUNCTION)"; fi
	@echo ""
	# Check and install memory_profiler
	@pip show memory_profiler > /dev/null 2>&1 || (echo "Installing memory_profiler..." && pip install memory_profiler psutil)
	# Clean old mprof files in profilers directory
	@rm -f $(PROFILERS_DIR)/mprofile_*.dat 2>/dev/null || true
	# Generate base filename
	$(eval BASENAME := $(subst /,_,$(FILE:.py=))$(if $(FUNCTION),_$(FUNCTION),)_$(TIMESTAMP))
	$(eval DAT_NAME := $(PROFILERS_DIR)/mprofile_$(BASENAME).dat)
	$(eval PNG_NAME := $(PROFILERS_DIR)/memory_profile_$(BASENAME).png)
	# Run memory profile
	@echo "Running memory profile..."
	@if [ -n "$(FUNCTION)" ]; then \
		echo "Running: mprof run -o $(DAT_NAME) pytest $(FILE)::$(FUNCTION) -s"; \
		mprof run -o $(DAT_NAME) pytest $(FILE)::$(FUNCTION) -s; \
	else \
		echo "Running: mprof run -o $(DAT_NAME) python3 $(FILE)"; \
		mprof run -o $(DAT_NAME) python3 $(FILE); \
	fi
	# Generate PNG plot
	@echo ""
	@echo "Generating memory plot..."
	@mprof plot -o $(PNG_NAME) $(DAT_NAME)
	@echo ""
	@echo "✓ Memory profile saved to: $(PNG_NAME)"
	@echo "✓ Data saved to: $(DAT_NAME)"
	@echo ""
	@echo "To view: eog $(PNG_NAME)  # or open with your image viewer"

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

# ============================================================
# UTILITY TARGETS
# ============================================================

# Clean profiling artifacts
profile-clean:
	@echo "Cleaning profiling artifacts..."
	# Remove all files in profilers directory
	@rm -rf $(PROFILERS_DIR)/* 2>/dev/null || true
	# Also clean any mprof files in root
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
	@echo "  # Clean all profiling files"
	@echo "  make profile-clean"