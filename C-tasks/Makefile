all: clean run

run:
	uv run python3 main.py --IconDataset ./IconDataset/png/ --Task3Dataset ./Task3Dataset/
	@feh -d output

run-opencv:
	uv run python3 main.py --IconDataset ./IconDataset/png/ --Task3Dataset ./Task3Dataset/ --ransac opencv
	@feh output

search:
	uv run python3 main.py --IconDataset ./IconDataset/png/ --GridSearch ./Task3Dataset/

clean:
	rm -f output/*
