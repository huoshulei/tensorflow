# # from pynput import keyboard
# #
# #
# # def on_press(key):
# #     try:
# #         print(key)
# #     except Exception as e:
# #         pass
# #
# #
# # def on_release(key):
# #     # print(key)
# #     if key == keyboard.Key.esc:
# #         return False
# #
# #
# # with keyboard.Listener(
# #         on_press=on_press,
# #         on_release=on_release,
# #         suppress=False) as listener:
# #     listener.join()
# import win32con
# import ctypes
# import ctypes.wintypes
# import threading
# import screenshot
#
# RUN = False
# EXIT = False
# user32 = ctypes.windll.user32
# id1 = 105
# id2 = 106
#
#
# class HotKey(threading.Thread):
#     # def __init__(self, screenshot):
#     #     self.screenshot = screenshot
#
#     def run(self):
#         global EXIT
#         global RUN
#         if not user32.RegisterHotKey(None, id1, win32con.MOD_CONTROL , win32con.VK_F1):
#             print("无法注册", id1)
#         if not user32.RegisterHotKey(None, id2, 0, win32con.VK_F10):
#             print("无法注册", id2)
#         try:
#             msg = ctypes.wintypes.MSG()
#             while True:
#                 if user32.GetMessageA(ctypes.byref(msg), None, 0, 0) != 0:
#                     if msg.message == win32con.WM_HOTKEY:
#                         if msg.wParam == id1:
#                             print("asdddddddddd")
#                             screenshot.buttonCaptureClick()
#                         elif msg.wParam == id2:
#                             pass
#                             return True
#                     user32.TranslateMessage(ctypes.byref(msg))
#                     user32.DispatchMessageA(ctypes.byref(msg))
#
#
#
#
#         except Exception as e:
#
#             pass
#         finally:
#             user32.UnregisterHotKey(None, id1)
#             user32.UnregisterHotKey(None, id2)
#
#
# hotkey = HotKey()
# hotkey.start()
