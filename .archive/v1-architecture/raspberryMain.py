import RPi.GPIO as GPIO
import subprocess

from utils.LARS_utils import TCPCommunication

### AMP Remote scancodes for ir-ctl
# Kenwood RC-80 IR codes (protocol: NEC)
# To send IR Blast for a specific code:
#   sudo ir-ctl -S nec:scancode
#
#   They can be chained together:
#   sudo ir-ctl -S nec:scancode1 -S nec:scancode2
#
#   You can also add repeats to simulate holding a button
#   sudo ir-ctl -S nec:scancode:n

CODES = {
    "Audio 1": 0xb881,
    "Audio 2": 0xb882,
    "Audio 3": 0xb883,
    "Audio 4": 0xb884,
    "Audio 5": 0xb885,
    "Audio 6": 0xb886,
    "Audio 7": 0xb887,
    "Audio 8": 0xb888,
    "Audio 9": 0xb889,
    "Audio 0": 0xb880,

    "TV/Video 1": 0x411,
    "TV/Video 2": 0x412,
    "TV/Video 3": 0x413,
    "TV/Video 4": 0x414,
    "TV/Video 5": 0x415,
    "TV/Video 6": 0x416,
    "TV/Video 7": 0x417,
    "TV/Video 8": 0x418,
    "TV/Video 9": 0x419,
    "TV/Video 0": 0x410,

    "+10": 0xb80d,

    "CD Pause/Play": 0xb8cb,
    "CD Stop": 0xb8c9,
    "CD Disc": 0xb808,
    "CD Skip Back": 0xb8ce,
    "CD Skip Forward": 0xb8cf,
    "CD Search Back": 0xb806,
    "CD Search Forward": 0xb807,

    "PHONO Back": 0xb8c1,
    "PHONO Stop": 0xb8c0,

    "TUNER Direct": 0xb89e,
    "TUNER A/B": 0xb89f,
    "TUNER P Scan": 0xb899,
    "TUNER FM": 0xb88f,
    "TUNER AM": 0xb88e,

    "TAPE A Rec": 0xb8d6,
    "TAPE A/TAPE B/VIDEO Skip Back": 0xb8da,
    "TAPE A/TAPE B/VIDEO Back": 0xb8d8,
    "TAPE A/TAPE B/VIDEO Forward": 0xb8d9,
    "TAPE A/TAPE B/VIDEO Skip Forward": 0xb8db,
    "TAPE B/VIDEO Rec": 0xb8de,
    "TAPE A/TAPE B/VIDEO Stop": 0xb8dd,
    "TAPE A/TAPE B/VIDEO Pause": 0xb8dc,

    "CD": 0xb892,
    "PHONO": 0xb890,
    "TUNER": 0xb891,
    "TAPE 1": 0xb894,
    "TAPE 2": 0xb895,
    "VIDEO 1": 0xb896,
    "VIDEO 2": 0xb893,
    "VIDEO 3": 0xb88a,

    "SYSTEM MEMORY 1": 0xb840,
    "SYSTEM MEMORY 2": 0xb841,

    "EQ.": 0xb8c5,
    "SURROUND": 0xb8d7,
    "MUTE": 0xb89c,
    "AUDIO POWER": 0xb89d,

    "TV POWER": 0x408,
    "VIDEO POWER": 0x1908,

    "TV CH. Up": 0x400,
    "TV CH. Down": 0x401,
    "TV": 0x409,
    "VIDEO": 0x190a,
    "TV VOL. Up": 0x402,
    "TV VOL. Down": 0x403,

    "REAR LEVEL Up": 0xb8c7,
    "REAR LEVEL Down": 0xb8c6,
    "MAIN VOL. Up": 0xb89b,
    "MAIN VOL. Down": 0xb89a,
}
###

class BinaryPinAction:
    def __init__(self, pins, on_voltage):
        """
            on_voltage - the setting to set the pin to turn this action on (HIGH or LOW)
            pins - an iterable of the pins controlled by this action 
        """

        self.pins = pins
        self.state = False # intialize to off
        self.on_voltage = on_voltage

        if self.on_voltage == GPIO.HIGH:
            self.off_voltage = GPIO.LOW
        else:
            self.off_voltage = GPIO.HIGH

        self.turn_on()
        
    def turn_on(self):
        self.on = True

        for pin in self.pins:
            GPIO.output(pin, self.on_voltage)
    
    def turn_off(self):
        self.on = False

        for pin in self.pins:
            GPIO.output(pin, self.off_voltage)

    def flip_state(self):
        for pin in self.pins:
            GPIO.output(pin, self.off_voltage if self.on else self.on_voltage)
        
        self.on = not self.on

class InfaRedAction:
    def __init__(self, protocol):
        self.protocol = protocol
        self.base_cmd = ["ir-ctl", "-S"]

    def blast(self, scancode_key):
        pulse = f"{self.protocol}:{CODES[scancode_key]}"
        cmd = self.base_cmd.append(pulse)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Error:", result.stderr)
        else:
            print("IR Signal sent.")
        

### Setup GPIO Pins ---
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM) # use physical board numbering

surge_protector_pin = 16

GPIO.setup(surge_protector_pin, GPIO.OUT)
###

### Setup Actions ---
#overhead_lamp = BinaryPinAction((overhead_lamp_pin,), GPIO.LOW)
#desk_lights = BinaryPinAction((desk_lights_pin,), GPIO.LOW)
master_switch = BinaryPinAction((surge_protector_pin,), GPIO.LOW)
#kenwood_amp = InfaRedAction("nec")
###

kw_to_action = {
    "lights": master_switch.flip_state,
    "lights on": master_switch.turn_on,
    "lights off": master_switch.turn_off
}

running = True

tcpCommunicator = TCPCommunication()

if __name__ == "__main__":
    print("Starting tcp server...")
    tcpCommunicator.openServer()

    try:
        while running:
            packet = tcpCommunicator.readFromClient()
            if packet is None or packet == "": continue
            print("Recieved packet:", packet)

            packet = packet.split(",")
            for i in packet:
                if i in list(kw_to_action.keys()):
                    action = kw_to_action[i]
                    action()

    except KeyboardInterrupt:
        print("\nStopping...")
        running = False
        tcpCommunicator.closeClientConnection()
        GPIO.cleanup()
