# serve_app.py
import http.server
import socketserver
import threading

PORT = 8080

def start_server():
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    # Put your HTML, CSS, JS files in the same directory
    start_server()