# syntax=docker/dockerfile:1.7
FROM node:22-alpine AS build
WORKDIR /workspace
ARG VITE_ICE_SERVERS_JSON="[]"
ENV VITE_ICE_SERVERS_JSON=$VITE_ICE_SERVERS_JSON
COPY package.json package-lock.json ./
COPY apps/web/package.json ./apps/web/package.json
RUN npm ci --workspace apps/web
COPY apps/web ./apps/web
RUN npm run build:web

FROM nginx:1.29-alpine
COPY apps/web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /workspace/apps/web/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=3s --retries=3 CMD wget -qO- http://127.0.0.1/healthz || exit 1
