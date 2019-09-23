import tty
import sys
import termios
from bluepy import btle
from pyparrot.Minidrone import Mambo
from bluetooth import *


def init_controller(addr=None):
	"""
	Initiate connection to Mambo via BLE and control it via keyboard (W, A, S, D, Q, E, F)
	:param addr: MAC address of Mambo to connect to, default: None
	:return: -
	"""
	if addr is None:
		mamboAddr = "D0:3A:58:76:E6:22"
	else:
		mamboAddr = addr
	mambo = Mambo(mamboAddr, use_wifi=False)

	print("trying to connect")
	success = mambo.connect(num_retries=3)
	print("connected: %s" % success)

	if success:
		orig_settings = termios.tcgetattr(sys.stdin)
		tty.setcbreak(sys.stdin)
		x = 0
		sending = False
		battery = mambo.sensors.battery
		print("Battery on take off:", battery)
		if not mambo.is_landed():
			mambo.safe_land(1)

		while x != chr(27):  # ESC

			x = sys.stdin.read(1)[0]
			# print("You pressed", x)
			# print(type(x))
			if x == 'f':  # if key 'q' is pressed
				if mambo.is_landed():
					mambo.ask_for_state_update()
					print("taking off!")
					mambo.safe_takeoff(1)
				else:
					print("landing with key")
					mambo.safe_land(1)
					break
			elif x == 'l':
				print("direct landing")
				mambo.safe_land(1)
			elif x == 'e':
				print("upwards!")
				mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=15, duration=1)
			elif x == 'q':
				print("downwards!")
				mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-15, duration=1)
			elif x == 'w':
				print("forwards!")
				mambo.fly_direct(roll=0, pitch=10, yaw=0, vertical_movement=0, duration=1)
			elif x == 's':
				print("backwards!")
				mambo.fly_direct(roll=0, pitch=-10, yaw=0, vertical_movement=0, duration=1)
			elif x == 'a':
				print("leftwards! :)")
				mambo.fly_direct(roll=0, pitch=0, yaw=-15, vertical_movement=0, duration=1)
			elif x == 'd':
				print("rightwards! :)")
				mambo.fly_direct(roll=0, pitch=0, yaw=15, vertical_movement=0, duration=1)
			elif x == 'c':
				if sending:
					print("stopping cam feed")
					sending = False
				else:
					print("activate cam feed")
					sending = True
			else:
				print("testing battery:")
				if mambo.sensors.battery != battery:
					battery = mambo.sensors.battery
					if battery < 7:
						print("landing because battery is low :)", battery)
						mambo.safe_land(1)
						break
					print("battery:", battery)

		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)
		print("disconnect")
		mambo.disconnect()


def list_devices():
	"""
	Search for available bluetooth devices until a Mambo is found
	:return: Name and MAC address of found Mambo as tuple or None tuple if interrupted by user
	"""
	# all_devices = []
	device = None
	mambo_found = False
	while not mambo_found:
		print("searching for devices..")
		try:
			nearby_devices = discover_devices(duration=10, lookup_names=True)
		except:
			break
		print("found %d devices" % len(nearby_devices))
		for addr, name in nearby_devices:
			print(" %s - %s" % (name, addr))
			# all_devices.append(tuple([name, addr]))
			if "Mambo" in name:
				print("Mambo found")
				mambo_found = True
				device = (name, addr)
				break
	if device is not None:
		return device
	else:
		return tuple([None, None])


	# print("found %d devices" % len(nearby_devices))
	# print("devices:", nearby_devices)

	# for name, addr in nearby_devices:
	# 	print(" %s - %s" % (addr, name))


"""
    print("Flying direct: going forward (positive pitch)")
    mambo.fly_direct(roll=0, pitch=50, yaw=0, vertical_movement=0, duration=1)

    print("Showing turning (in place) using turn_degrees")
    mambo.turn_degrees(90)
    mambo.smart_sleep(2)
    mambo.turn_degrees(-90)
    mambo.smart_sleep(2)

    print("Flying direct: yaw")
    mambo.fly_direct(roll=0, pitch=0, yaw=50, vertical_movement=0, duration=1)

    print("Flying direct: going backwards (negative pitch)")
    mambo.fly_direct(roll=0, pitch=-50, yaw=0, vertical_movement=0, duration=0.5)

    print("Flying direct: roll")
    mambo.fly_direct(roll=50, pitch=0, yaw=0, vertical_movement=0, duration=1)

    print("Flying direct: going up")
    mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=50, duration=1)

    print("Flying direct: going around in a circle (yes you can mix roll, pitch, yaw in one command!)")
    mambo.fly_direct(roll=25, pitch=0, yaw=50, vertical_movement=0, duration=3)
"""