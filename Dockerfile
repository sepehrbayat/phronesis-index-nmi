FROM python:3.11-slim

# Install LaTeX (minimal but sufficient for the manuscript)
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-science \
    texlive-bibtex-extra \
    cm-super \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir numpy scipy networkx matplotlib

# Copy the entire repo
WORKDIR /repo
COPY . /repo/

# Make scripts executable
RUN chmod +x scripts/*.sh

# Default: run everything (experiments + LaTeX + validation)
CMD ["bash", "scripts/docker_run_all.sh"]
