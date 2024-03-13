from gps import gps, WATCH_ENABLE, WATCH_NEWSTYLE
import csv
import ntplib
from datetime import datetime, timezone

filepath = "./gps_readings.csv"

def getPositionData():
    session = gps(mode=WATCH_ENABLE)
    session.stream(WATCH_ENABLE | WATCH_NEWSTYLE)

    c = ntplib.NTPClient()
    response = c.request('pool.ntp.org', version=3)
    response.offset
    print(datetime.fromtimestamp(response.tx_time, timezone.utc))

    positions = []
    totalcount = 0

    while True:
        try:
            report = session.next()
            # For a list of all supported classes and fields refer to:
            # https://gpsd.gitlab.io/gpsd/gpsd_json.html
            if report['class'] == 'TPV':
                latitude = getattr(report, 'lat', "Unknown")
                longitude = getattr(report, 'lon', "Unknown")
                time = getattr(report, 'time', "Unknown")
                print(report)
                print("Your position: lon = " +
                      str(longitude) + ", lat = " +
                      str(latitude) + ", time = " +
                      str(time))
                positions.append((longitude, latitude))
                if len(positions) >= 10:
                    with open(filepath, 'a', newline='') as f:
                        writer = csv.writer(f, delimiter='\t')
                        writer.writerow(positions)
                        positions = []
                    totalcount += 10
                    print("Total gps readings: ", totalcount)
        except KeyError:
            pass
        except KeyboardInterrupt:
            quit()
        except StopIteration:
            session = None
            print("GPSD has terminated")


if __name__ == "__main__":
    getPositionData()
