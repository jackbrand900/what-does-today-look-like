import webscraper as ws

webscraper = ws.Webscraper(safe_mode=True)
print(webscraper.generate_prompt())