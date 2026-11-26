FROM python:3.11-slim

WORKDIR /app

# git: xarray-ms is pinned to a commit of its write-support branch until that
# alpha is on PyPI (see pyproject.toml), and uv needs git to fetch it.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package installation
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /usr/local/bin/uv

# Copy package files (LICENSE is required by pyproject's license-files glob)
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install package with full dependencies.
# python-casacore, numba and dask-ms all ship manylinux wheels, so no apt-get
# of casacore/llvm is needed on this base image.
# --prerelease=allow: the xarray-ms/arcae write-support line is prerelease-only.
RUN uv pip install --system --no-cache --prerelease=allow ".[full,web]"

# Make CLI available
CMD ["surfvis", "--help"]
