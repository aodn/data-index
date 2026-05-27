# Define all abstract commands here
.PHONY: test-image init

init:
	@echo "Installing pre-commit..."
	uv pip install pre-commit
	@echo "Configuring git hook stages..."
	uv run pre-commit install --hook-type pre-commit
	uv run pre-commit install --hook-type pre-push
	@echo "✓ Setup complete! Hooks will run on commit and push."

test-image:
	rm -rf dist/
	uv build
	docker build --target test \
	--no-cache-filter test --no-cache-filter app \
	-o type=cacheonly .
