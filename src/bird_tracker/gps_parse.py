from gps import gps, WATCH_ENABLE, WATCH_NEWSTYLE


def getPositionData():
    session = gps(mode=WATCH_ENABLE)
    session.stream(WATCH_ENABLE | WATCH_NEWSTYLE)

    while True:
        try:
            report = session.next()
            # For a list of all supported classes and fields refer to:
            # https://gpsd.gitlab.io/gpsd/gpsd_json.html
            if report['class'] == 'TPV':
                latitude = getattr(report, 'lat', "Unknown")
                longitude = getattr(report, 'lon', "Unknown")
                print("Your position: lon = " +
                      str(longitude) + ", lat = " + str(latitude))
                return (longitude, latitude)
        except KeyError:
            pass
        except KeyboardInterrupt:
            quit()
        except StopIteration:
            session = None
            print("GPSD has terminated")


if __name__ == "__main__":
    getPositionData()
