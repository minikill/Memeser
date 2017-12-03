import sys
import telebot
import cv2 as cv
import numpy as np
from math import atan
from math import degrees
from telebot import types
sys.path.append('config/')
import config

face_cascade = cv.CascadeClassifier('config/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('config/haarcascade_eye.xml')
bot = telebot.TeleBot(config.token)

@bot.message_handler(commands=['help'])
def send_help_message(message):
    #print(message)
    bot.send_message(message.chat.id, 'Send me a picture with your face, and choose memes!')

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    users_photo = bot.download_file(file_info.file_path)
    path = 'pictures/user_pictures/' + str(message.from_user.id) + '.jpg' #change extention by regex
    f = open(path, 'w')
    f.write(users_photo)
    f.close()
    markup = types.ReplyKeyboardMarkup()
    markup.row('Deal With It', 'Pepe Frog')
    markup.row('test2', 'test3')
    bot.send_message(message.chat.id, "Choose one memes:", reply_markup=markup)

@bot.message_handler(content_types=['text'])
def handle_memes(message):
    if message.text == 'Deal With It':
        deal_with_it(message)
    if message.text == 'Pepe Frog':
        pepe_frog(message)
    if message.text == 'test2':
        #test2()
        pass
    if message.text == 'test3':
        #test3()
        pass

def deal_with_it(message):
    path = 'pictures/user_pictures/' + str(message.from_user.id) + '.jpg' #change extention by regex
    result = 'pictures/result/' + str(message.from_user.id) + '.jpg' #change extention by regex
    image = cv.imread(path, 1)
    gray = cv.imread(path, 0)
    glass = cv.imread('pictures/parts/glass.jpg')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #(ex,ey,ew,eh)
        '''
        first_middle_x = (eyes[0][0]*2+eyes[0][2])/2
        first_middle_y = (eyes[0][1]*2+eyes[0][3])/2
        second_middle_x = (eyes[1][0]*2+eyes[1][2])/2
        second_middle_y = (eyes[1][1]*2+eyes[1][3])/2
        angle = int(degrees(atan((second_middle_x-first_middle_x)/(second_middle_y-first_middle_y))))
        rot_mat = cv.getRotationMatrix2D((glass.shape[0]/2,glass.shape[1]/2),angle,1)
        glass = cv.warpAffine(glass, rot_mat, glass.shape)
        '''

        glass = cv.resize(glass, (eyes[1][0]+eyes[1][2]-eyes[0][0], eyes[0][3]))

        y1, y2 = y, y + glass.shape[0]
        x1, x2 = x, x + glass.shape[1]

        alpha_glass = glass[:, :, 1] / 255.0 + glass[:, :, 2] / 255.0
        alpha_image = 1.0 - alpha_glass

        for c in range(0, 3):
            image[y1:y2, x1:x2, c] = (alpha_glass * glass[:, :, c] + alpha_image * image[y1:y2, x1:x2, c])


    cv.imwrite(result,image)
    bot.send_photo(message.chat.id, open(result, 'r'))

def pepe_frog(message):
    path = 'pictures/user_pictures/' + str(message.from_user.id) + '.jpg' #change extention by regex
    result = 'pictures/result/' + str(message.from_user.id) + '.jpg' #change extention by regex
    image = cv.imread(path, 1)
    gray = cv.imread(path, 0)
    pepe = cv.imread('pictures/parts/pepe.jpg')
    alpha_pepe = cv.imread('pictures/parts/pepe_alpha.jpg')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        pepe = cv.resize(pepe, (w, h))
        alpha_pepe = cv.resize(alpha_pepe, (w, h))

        y1, y2 = y, y + pepe.shape[0]
        x1, x2 = x, x + pepe.shape[1]

        frame = image[y1:y2, x1:x2]
        pepe = pepe.astype(float)
        frame = frame.astype(float)
        alpha_pepe = alpha_pepe.astype(float)/255
        pepe = cv.multiply(alpha_pepe, pepe)
        frame = cv.multiply(1.0 - alpha_pepe, frame)

        image[y1:y2, x1:x2] = cv.add(pepe, frame)

    cv.imwrite(result,image)
    bot.send_photo(message.chat.id, open(result, 'r'))

while True:
    try:
        bot.polling(none_stop=True)
    except Exception as e:
        print(e)
