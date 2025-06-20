FROM nginx:alpine

# Copy the static files
COPY . /usr/share/nginx/html

# Create nginx configuration
RUN echo 'events { worker_connections 1024; } \
http { \
    include /etc/nginx/mime.types; \
    default_type application/octet-stream; \
    sendfile on; \
    keepalive_timeout 65; \
    server { \
        listen 4002; \
        server_name localhost; \
        location / { \
            root /usr/share/nginx/html; \
            index index.html; \
            try_files $uri $uri/ /index.html; \
        } \
        location /health { \
            access_log off; \
            return 200 "healthy\n"; \
            add_header Content-Type text/plain; \
        } \
    } \
}' > /etc/nginx/nginx.conf

# Expose port
EXPOSE 4002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:4002/health || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]