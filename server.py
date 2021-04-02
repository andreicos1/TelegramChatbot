import json
from chatbot import telegram_chatbot
import chatbot_AI

update_id=None
intents = json.loads(open('intents.json').read()) #open the intents file


def make_reply(msg):
    reply = None
    if msg is not None:
        ints = chatbot_AI.predict_class(message)  # get the intent
        reply= chatbot_AI.get_response(ints, intents)
    return reply

bot = telegram_chatbot('config.cfg')

while True:
    print('...')
    updates = bot.get_updates(offset=update_id)
    updates = updates['result']
    if updates:
        print(updates)
        for item in updates:
            update_id = item['update_id']
            try:
                message = item['message']['text']
            except:
                message=None
            from_ = item['message']['from']['id']
            reply = make_reply(message)
            bot.send_message(reply, from_)