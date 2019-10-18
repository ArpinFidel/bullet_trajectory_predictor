import requests
from random import randint
import csv
from config import *

def call_api(angle, velocity, w_angle, w_speed):
    req = requests.post("https://wrapapi.com/use/Arpin/bullet_trajectory_hornady/calculator/0.0.2", json={
        "angle": angle,
        "velocity": velocity,
        "wind_angle": w_angle,
        "wind_speed": w_speed,
        "wrapAPIKey": wrap_api_key
    })
    print(angle, velocity, w_angle, w_speed)
    for r in req.json()['data']['output2']: print(r)
    return [(r['Range'], r['Velocity'], r['Path'], r['WindMil']) for r in req.json()['data']['output2']]

# angle     -90 - 90
# velocity   30 - 1400
# wind a      0 - 359
# wind s      0 - 115

with open('bullet.csv', 'a') as csv_f:
    fields = ['angle', 'velocity', 'wind_angle', 'wind_speed', 'range', 'windmil']
    writer = csv.DictWriter(csv_f, fieldnames=fields)

    for _ in range(1000):
        angle = randint(-90, 90)
        velocity = randint(30, 1400)
        wind_angle = randint(0, 359)
        wind_speed = randint(0, 115)
    
        req = call_api(angle, velocity, wind_angle, wind_speed)[1:]
        
        r = None
        if req[-1][0] != 0:
            for i in range(1, len(req)):
                if req[i][2] < req[i-1][2]:
                    r = min(req[i-1:], key=(lambda x: abs(x[2])))
                    break
        else: r = (0, 0, 0, req[0][3])
        if not r: r = (req[-1][0]+req[-1][1], 0, 0, req[-1][3])

        print(r)

        writer.writerow({f:x for f,x in zip(fields, [angle, velocity, wind_angle, wind_speed]+[r[0], r[3]])})
