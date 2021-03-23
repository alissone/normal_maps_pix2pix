from matplotlib.backends.backend_agg import FigureCanvasAgg
from discord_webhook import DiscordWebhook, DiscordEmbed
from matplotlib.figure import Figure
import numpy as np
import cv2
import http.client
import configparser

config = configparser.ConfigParser()
config.read("secrets.ini")
webhookurl = config.get("discord", "webhook_url")


def send_message(message):
    formdata = "------:::BOUNDARY:::\r\nContent-Disposition: form-data; name=\"content\"\r\n\r\n" + \
        message + "\r\n------:::BOUNDARY:::--"
    connection = http.client.HTTPSConnection("discordapp.com")
    connection.request("POST", webhookurl, formdata, {
        'content-type': "multipart/form-data; boundary=----:::BOUNDARY:::",
        'cache-control': "no-cache",
    })
    response = connection.getresponse()
    result = response.read()
    return result.decode("utf-8")


def send_image(*images):
    webhook = DiscordWebhook(url=webhookurl, username="Google Colab")
    image_path = "image.jpg"

    for idx, image in enumerate(images):
        cv2.imwrite(image_path, image)
        with open(image_path, "rb") as f:
            webhook.add_file(
                file=f.read(), filename="image_{}.jpg".format(idx))
    return webhook.execute()


def send_file(*files):
    webhook = DiscordWebhook(url=webhookurl, username="Google Colab")

    for idx, filename in enumerate(files):
        with open(filename, "rb") as f:
            webhook.add_file(file=f.read(), filename=filename.split("/")[-1])

    return webhook.execute()


def plot_to_img(plot_data):
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)

    ax = fig.add_subplot(111)
    ax.plot(plot_data)

    canvas.draw()
    buf = canvas.buffer_rgba()
    return np.asarray(buf)
