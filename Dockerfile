# Dockerfile (UPDATED)

FROM mambaorg/micromamba:latest

WORKDIR /app

ENV MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER credit_risk_env.yml /app/
RUN micromamba install -y -n base -f /app/credit_risk_env.yml && \
    micromamba clean -afy --quiet

# Copy application code
COPY --chown=$MAMBA_USER:$MAMBA_USER ./src /app/src

# ✨ NEW: Copy model directory
COPY --chown=$MAMBA_USER:$MAMBA_USER ./models /app/models

# ✨ NEW: Copy raw data (instead of dashboard_data.csv)
COPY --chown=$MAMBA_USER:$MAMBA_USER ./data/application_test.csv /app/data/application_test.csv

# Keep SHAP pre-computed (77 MB, no need to recompute)
COPY --chown=$MAMBA_USER:$MAMBA_USER ./shap /app/shap

# Keep pre-computed histograms for distribution plots
COPY --chown=$MAMBA_USER:$MAMBA_USER ./plots /app/plots

# Copy tests
COPY --chown=$MAMBA_USER:$MAMBA_USER ./tests /app/tests

# Copy entrypoint
COPY --chown=$MAMBA_USER:$MAMBA_USER --chmod=755 entrypoint.sh /app/

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
