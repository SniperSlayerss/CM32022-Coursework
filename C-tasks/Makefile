PYTHON := $(shell if command -v uv >/dev/null 2>&1; then echo "uv run python3"; else echo "python3"; fi)

c1:
	$(PYTHON) main.py --Task1Dataset ./Task1Dataset/

c2:
	$(PYTHON) main.py --IconDataset ./IconDataset/png/ --Task2Dataset ./Task2Dataset/

c3:
	$(PYTHON) main.py --IconDataset ./IconDataset/png/ --Task3Dataset ./Task3Dataset/

c3-search:
	$(PYTHON) main.py --IconDataset ./IconDataset/png/ --GridSearch ./Task3Dataset/

clean:
	rm -f output/*
