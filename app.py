from flask import Flask, render_template, Response, jsonify
import cv2

app = Flask(__name__)

# Initialize variables
avg = None
video = cv2.VideoCapture("people-capture.mp4")
xvalues = []
motion = []
count1 = 0
count2 = 0

def find_majority(k):
    myMap = {}
    maximum = ('', 0)  # (occurring element, occurrences)
    for n in k:
        if n in myMap:
            myMap[n] += 1
        else:
            myMap[n] = 1
        
        # Keep track of maximum on the go
        if myMap[n] > maximum[1]:
            maximum = (n, myMap[n])

    return maximum

def generate_frames():
    global avg, xvalues, motion, count1, count2

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Preprocess the frame
        frame = cv2.resize(frame, (500, 375))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if avg is None:
            print("[INFO] Starting background model...")
            avg = gray.astype("float")
            continue

        # Update the running average of the background
        cv2.accumulateWeighted(gray, avg, 0.5)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 5000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            xvalues.append(x)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        no_x = len(xvalues)
        
        if no_x > 2:
            difference = xvalues[no_x - 1] - xvalues[no_x - 2]
            motion.append(1 if difference > 0 else 0)

        if not contours:
            if no_x > 5:
                val, times = find_majority(motion)
                if val == 1 and times >= 15:
                    count1 += 1
                else:
                    count2 += 1
                    
            xvalues = []
            motion = []

        cv2.line(frame, (200, 0), (200, 375), (0, 255, 0), 2)
        cv2.line(frame, (300, 0), (300, 375), (0, 255, 0), 2)
        cv2.putText(frame, "In: {}".format(count1), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Out: {}".format(count2), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Create a Flask-compatible image to show in the template
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the processed frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count', methods=['GET'])
def count():
    in_count = count1
    out_count = count2
    return jsonify(in_count=in_count, out_count=out_count)

if __name__ == '__main__':
    app.run(debug=True)
