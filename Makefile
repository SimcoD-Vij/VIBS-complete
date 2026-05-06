.PHONY: up down logs build restart shell-backend shell-worker clean

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

logs-backend:
	docker-compose logs -f backend

logs-worker:
	docker-compose logs -f worker

build:
	docker-compose build --no-cache

restart:
	docker-compose restart backend worker

shell-backend:
	docker-compose exec backend bash

shell-worker:
	docker-compose exec worker bash

shell-db:
	docker-compose exec postgres psql -U vibs_user -d vibs

clean:
	docker-compose down -v
	rm -rf audio_files/*.wav audio_files/*.webm

status:
	docker-compose ps
	@echo ""
	@curl -s http://localhost:8000/ | python3 -m json.tool 2>/dev/null || echo "Backend not responding"
