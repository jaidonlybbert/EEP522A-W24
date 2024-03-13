import math
from typing import Dict


def gps_conv_factors_deg_to_meters(degrees_latitude: float) -> Dict:
	# Compute lengths of degrees at specific latitude

	# Convert latitude to radians
	lat = math.radians(degrees_latitude)

	# Set up "Constants"
	m1 = 111132.92		# latitude calculation term 1
	m2 = -559.82		# latitude calculation term 2
	m3 = 1.175			# latitude calculation term 3
	m4 = -0.0023		# latitude calculation term 4
	p1 = 111412.84		# longitude calculation term 1
	p2 = -93.5			# longitude calculation term 2
	p3 = 0.118			# longitude calculation term 3

	# Calculate the length of a degree of latitude and longitude in meters
	latlen = m1 + (m2 * math.cos(2 * lat)) + (m3 * math.cos(4 * lat)) +\
		(m4 * math.cos(6 * lat))

	longlen = (p1 * math.cos(lat)) + (p2 * math.cos(3 * lat)) +\
		(p3 * math.cos(5 * lat))

	return {"meters_per_degree": {"latitude": latlen, "longitude": longlen}}

    
if __name__ == "__main__":
	gps_conversion = gps_conv_factors_deg_to_meters(45.0)
	print(gps_conversion["meters_per_degree"])
