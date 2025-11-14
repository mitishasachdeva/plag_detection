# Use a standard Python 3.10 image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file (it's in the root)
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy *only* the backend folder into our app directory
COPY ./backend .

# Tell Docker the app runs on port 7860
EXPOSE 7860

# Command to run the app (app.py is now in the root of /app)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]