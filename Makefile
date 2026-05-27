# Define all abstract commands here
.PHONY: test-image

test-image:
	rm -rf dist/
	uv build
	docker build --target test \
	--no-cache-filter test --no-cache-filter app \
	-o type=cacheonly .