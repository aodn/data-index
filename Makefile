# Define all abstract commands here
.PHONY: test test-image init lint

# Lint
lint:
	uvx ruff check --fix && uvx ruff format

# Run tests
test:
	uv run pytest tests/ -v

# Initialise pre-commit
init:
	@echo "Installing pre-commit..."
	uv pip install pre-commit
	@echo "Configuring git hook stages..."
	uv run pre-commit install --hook-type pre-commit
	uv run pre-commit install --hook-type pre-push
	@echo "✓ Setup complete! Hooks will run on commit and push."

# Build the package and test on the docker image
test-image:
	@echo "Building distribution"
	rm -rf dist/
	uv build
	@echo "Building and testing image on docker"
	docker build --target test \
	--no-cache-filter test --no-cache-filter app \
	--progress=plain \
	-o type=cacheonly .