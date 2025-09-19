# Use a lightweight micromamba image as a base
FROM mambaorg/micromamba:latest

# Set the working directory inside the container
WORKDIR /app

# Set environment variables for micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy the environment file and install dependencies first for better caching
COPY --chown=$MAMBA_USER:$MAMBA_USER credit_risk_env.yml /app/
RUN micromamba install -y -n base -f /app/credit_risk_env.yml && \
    micromamba clean -afy --quiet

# Copy the application code and data
# Note: We are NO LONGER copying the 'models' directory
COPY --chown=$MAMBA_USER:$MAMBA_USER ./src /app/src
COPY --chown=$MAMBA_USER:$MAMBA_USER ./data/dashboard_data.csv /app/data/dashboard_data.csv
COPY --chown=$MAMBA_USER:$MAMBA_USER ./shap /app/shap
COPY --chown=$MAMBA_USER:$MAMBA_USER ./plots /app/plots

# Copy test files (optional, can be removed for a smaller production image)
COPY --chown=$MAMBA_USER:$MAMBA_USER ./tests /app/tests
COPY --chown=$MAMBA_USER:$MAMBA_USER ./pytest.ini /app/pytest.ini

# Copy and set executable permissions for the entrypoint script
COPY --chown=$MAMBA_USER:$MAMBA_USER --chmod=755 entrypoint.sh /app/

# Expose the public port for the Streamlit app
EXPOSE 7860

# Set the entrypoint to run the application
ENTRYPOINT ["/app/entrypoint.sh"]
