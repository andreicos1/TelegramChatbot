import requests
import json
import configparser as cfg


class telegram_chatbot():
    def __init__(self, config):
        self.token = self.read_token_from_config_file(config)
        self.base = f"https://api.telegram.org/bot{self.token}"

    def get_updates(self, offset=None):
        url = self.base + "/getUpdates?timeout=100"
        if offset:
            url = url + f"&offset={offset + 1}"
        r = requests.get(url)
        content = r.content.decode("utf-8")
        print(url)
        return json.loads(content)

    def send_message(self, msg, chat_id):
        url = self.base + f"/sendMessage?text={msg}&chat_id={chat_id}"
        if msg is not None:
            requests.get(url)

    def read_token_from_config_file(self, config):
        parser = cfg.ConfigParser()
        parser.read(config)
        return parser.get('creds', 'token')
