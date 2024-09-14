import sensor
import time
import ml
import display

sensor.reset()  # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)  # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QQVGA2)  # Special 128x160 framesize for LCD Shield.

sensor.skip_frames(time=2000)  # Let the camera adjust.



model = ml.Model("model_quant.tflite", load_to_fb=True)

labels = [line.rstrip("\n") for line in open("labels.txt")]

clock = time.clock()
lcd = display.SPIDisplay()


while True:
    clock.tick()

    img = sensor.snapshot()


    # This combines the labels and confidence values into a list of tuples
    # and then sorts that list by the confidence values.
    sorted_list = sorted(
        zip(labels, model.predict([img])[0].flatten().tolist()), key=lambda x: x[1], reverse=True
    )

    x = []
    for i in range(len(sorted_list)):
        x.append(sorted_list[i][1])

    x_max = max(x)
    x_min = min(x)

    for i in range(len(x)):
        x[i] = (x[i]-x_min)/(x_max-x_min)


    x_max = max(x)
    print(x_max)

    for i in range(len(x)):
        if x[i] == x_max:
            img.draw_rectangle(0,0,128,10,(0,0,0),fill=True)
            img.draw_string(2,0,"%s = %s" % (sorted_list[i][0], str(round(sorted_list[i][1],2))), (255,255,255))
            print("%s = %s" % (sorted_list[i][0], str(round(sorted_list[i][1],2))))


    lcd.write(img)
    print(clock.fps(), "fps")
