import ctypes

# Constants
HWND_BROADCAST = 0xFFFF
WM_SYSCOMMAND = 0x0112
SC_MONITORPOWER = 0xF170
MONITOR_ON = -1
MONITOR_OFF = 2

last_display_state = None

def display_off():
    ctypes.windll.user32.PostMessageW(HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, MONITOR_OFF)

def display_on():
    # move the mouse to wake the display
    ctypes.windll.user32.mouse_event(1, 1, 0, 0)
    ctypes.windll.user32.mouse_event(1, -1, 0, 0)

def display_toggle():
    global last_display_state
    if last_display_state is None or last_display_state == False:
        display_on()
        last_display_state = True
    else:
        display_off()
        last_display_state = False