"""REST-сервис классификации обращений.

Запуск:   python server.py [--mode hybrid|classic] [--port 8000]
Запросы:  GET  /health    — состояние сервиса
          POST /classify  — тело JSON {"text": "текст обращения", "login": "логин клиента"}
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import classifier


class Handler(BaseHTTPRequestHandler):
    mode = 'hybrid'

    def send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != '/health':
            return self.send_json(404, {'error': 'неизвестный адрес; доступны GET /health и POST /classify'})
        ollama_ok, problem = classifier.ollama_status()
        self.send_json(200, {'status': 'ok', 'mode': self.mode,
                             'ollama': ollama_ok if self.mode == 'hybrid' else 'не нужна в режиме classic',
                             'problem': problem if self.mode == 'hybrid' else ''})

    def do_POST(self):
        if self.path != '/classify':
            return self.send_json(404, {'error': 'неизвестный адрес; доступны GET /health и POST /classify'})
        try:
            request = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
        except json.JSONDecodeError:
            return self.send_json(400, {'error': 'тело запроса должно быть JSON'})
        text, login = request.get('text'), request.get('login')
        if not isinstance(text, str) or not text.strip() or not isinstance(login, str):
            return self.send_json(400, {'error': 'нужны поля "text" (непустая строка) и "login" (строка)'})
        try:
            self.send_json(200, classifier.classify(text, login, self.mode))
        except Exception as error:  # например, Ollama остановлена во время работы
            self.send_json(500, {'error': f'ошибка классификации: {error}'})


def main():
    parser = argparse.ArgumentParser(description='REST-сервис классификации обращений RUDATA')
    parser.add_argument('--mode', choices=classifier.MODES, default='hybrid')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    if args.mode == 'hybrid':
        ollama_ok, problem = classifier.ollama_status()
        if not ollama_ok:
            print(f'Режим hybrid недоступен: {problem}. Запустите Ollama или используйте --mode classic')
            return

    Handler.mode = args.mode
    print(f'Сервис запущен: http://localhost:{args.port}  (режим {args.mode}). Остановить — Ctrl+C')
    try:
        HTTPServer(('localhost', args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print('Сервис остановлен')


if __name__ == '__main__':
    main()
