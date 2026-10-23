.PHONY: all demo test clean

all: test demo

demo:
	python3 demo.py

test:
	python3 -m unittest discover -s tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
