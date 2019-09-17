import tty
import sys
import termios
from pyparrot.Minidrone import Mambo


def init_controller():
	mamboAddr = "D0:3A:58:76:E6:22"
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
			print("testing battery:")
			if mambo.sensors.battery != battery:
				battery = mambo.sensors.battery
				if battery < 7:
					print("landing because battery is low :)", battery)
					mambo.safe_land(1)
					break
				print("battery:", battery)
		# print("z orientation:", mambo.sensors.get_estimated_z_orientation())

		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)
		print("disconnect")
		mambo.disconnect()
