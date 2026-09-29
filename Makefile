# Synthetic-data pipeline: CAD parts (STEP) -> instance-segmentation
# training data -> trained YOLO26-seg model.
#
# Every target goes through src/main.py, which reads the Isaac Sim path from
# pyproject.toml and dispatches each stage to the right Python interpreter -
# so that path lives in exactly one place, not duplicated here.
#
# Written to run unchanged under both shells GNU Make may use on Windows:
# cmd.exe (when launched from PowerShell/cmd) and sh (from Git Bash). That
# means no shell built-ins like rm, and no quotes/parentheses in echo.
#
# Override INPUT_DIR/OUTPUT_DIR on the command line, e.g.:
#   make all INPUT_DIR=my_parts OUTPUT_DIR=out

INPUT_DIR ?= examples/fasteners
OUTPUT_DIR ?= workspace

RUN := uv run python src/main.py --input-dir $(INPUT_DIR) --output-dir $(OUTPUT_DIR)

.PHONY: help setup convert generate dataset train all clean

help:
	@echo Targets: setup convert generate dataset train all clean
	@echo Vars: INPUT_DIR=$(INPUT_DIR) OUTPUT_DIR=$(OUTPUT_DIR)

setup:
	uv sync

convert:
	$(RUN) --stage convert

generate:
	$(RUN) --stage generate

dataset:
	$(RUN) --stage dataset

train:
	$(RUN) --stage train

all:
	$(RUN)

clean:
	uv run python -c "import shutil; shutil.rmtree(r'$(OUTPUT_DIR)', ignore_errors=True)"
